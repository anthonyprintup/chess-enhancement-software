from __future__ import annotations

import asyncio
import functools
import json
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, AsyncIterator, Pattern, TypeVar, TypeAlias

import chess
from chess.engine import Limit as ChessEngineLimit, UciProtocol as ChessEngine, popen_uci as open_chess_engine
from playwright.async_api import Browser, BrowserContext, ElementHandle, JSHandle, Locator, Page, Playwright, \
    APIResponse, Route, WebSocket, async_playwright

BINARIES_PATH: Path = Path.cwd() / "binaries"
STOCKFISH_PATH: Path = BINARIES_PATH / "stockfish"
STOCKFISH_ENGINE_PATH: Path = STOCKFISH_PATH / "stockfish-windows-2022-x86-64-avx2.exe"
STOCKFISH_SETTINGS_PATH: Path = STOCKFISH_PATH / "settings.json"
SCRIPTS_PATH: Path = Path(__file__).resolve().parent / "scripts"


BrowserHandlerType = TypeVar("BrowserHandlerType", bound="BrowserHandler")
AnalysisResultType: TypeAlias = tuple[chess.engine.AnalysisResult, chess.engine.BestMove]


@cache
def read_scripts() -> dict[str, str]:
    scripts: dict[str, str] = {}
    for path in SCRIPTS_PATH.iterdir():
        if not path.is_file() or not path.suffix == ".js":
            continue
        scripts[path.name] = path.read_text()
    return scripts


@dataclass
class BrowserHandler(ABC):
    browser: Browser
    browser_context: BrowserContext

    @staticmethod
    @abstractmethod
    async def create(playwright: Playwright) -> AsyncIterator[BrowserHandlerType]:
        raise NotImplementedError


@dataclass
class UserSettings:
    troll_opponents: bool = False
    auto_move: bool = False
    abuse_premoves: bool = False


@dataclass(frozen=True)
class EngineControlScheme:
    condition: str
    operation: str


@dataclass
class Round:
    owner_page: Page
    identifier: str
    player_color: chess.Color
    chess_board: chess.Board
    transport: asyncio.SubprocessTransport
    chess_engine: ChessEngine
    engine_control_schemes: list[EngineControlScheme]
    chess_engine_limits: ChessEngineLimit = field(default_factory=ChessEngineLimit)
    user_settings: UserSettings = field(default_factory=UserSettings)
    # Secret variables
    _cp_scores: list[tuple[int, int]] = field(default_factory=list)
    _chess_engine_analysis_task: asyncio.Task | None = None
    _takeback_offer_origin: chess.Color | None = None
    # UI variables
    _canvas_height_offset: int = 44  # how much to expand the height of the canvas by (for the engine data)
    _shadow_root: JSHandle | None = None
    _current_match_data: dict = field(default_factory=dict)
    # Locks
    _on_move_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @staticmethod
    async def create(page: Page, identifier: str, player_color: chess.Color,
                     move_data: list[dict], time_increment: float) -> Round:
        # Setup the chess board
        chess_board: chess.Board = chess.Board(fen=move_data[0]["fen"])
        for move in move_data[1:]:
            chess_board.push_uci(move["uci"])

        # Read the preferred settings from disk (note: do not cache)
        settings: dict = json.loads(STOCKFISH_SETTINGS_PATH.read_text())

        # Parse the engine control schemes
        engine_control_schemes: list[EngineControlScheme] = []
        for condition, operation in ((engine_control_scheme["condition"], engine_control_scheme["operation"])
                                     for engine_control_scheme in settings["engine-control-schemes"]):
            engine_control_schemes.append(EngineControlScheme(condition=condition, operation=operation))

        # Spawn an engine instance and configure the engine
        transport, engine = await open_chess_engine(str(STOCKFISH_ENGINE_PATH))
        await engine.configure(options=settings["uci-settings"])

        # Create a round instance
        round_instance: Round = Round(owner_page=page, identifier=identifier, player_color=player_color,
                                      chess_board=chess_board, transport=transport, chess_engine=engine,
                                      engine_control_schemes=engine_control_schemes)
        # Setup time increments
        round_instance.chess_engine_limits.white_inc = time_increment
        round_instance.chess_engine_limits.black_inc = time_increment

        # Configure the settings
        round_instance.update_settings(settings={
            "engine-depth": settings["engine-limits"]["depth"]
        } | settings["fun-settings"])

        # Create the shadow dom
        await round_instance.create_shadow_root()

        # Queue engine analysis if the user begins the match
        if chess_board.turn == player_color:
            # Wait for the engine to be ready
            await engine.ping()
            round_instance.queue_engine_analysis()

        # Return the round instance
        return round_instance

    def update_settings(self, settings: dict) -> None:
        for key, value in settings.items():
            if key == "engine-depth":
                self.chess_engine_limits.depth = value
            elif key == "troll-opponents":
                self.user_settings.troll_opponents = value
            elif key == "auto-move":
                self.user_settings.auto_move = value
            elif key == "abuse-premoves":
                self.user_settings.abuse_premoves = value

    @property
    def scripts(self) -> dict[str, str]:
        return read_scripts()

    def queue_engine_analysis(self) -> None:
        # Cancel any pending tasks
        self.cancel_engine_analysis()

        # Check if the engine is ready
        self._chess_engine_analysis_task = asyncio.create_task(self.find_best_move(), name="Best move task")
        self._chess_engine_analysis_task.add_done_callback(self.on_engine_analysis_finished)

    def cancel_engine_analysis(self) -> None:
        if self._chess_engine_analysis_task is None or self._chess_engine_analysis_task.done():
            return
        self._chess_engine_analysis_task.cancel(msg="New engine analysis task queued.")
        self._chess_engine_analysis_task = None

    async def find_best_move(self) -> AnalysisResultType:
        analysis: chess.engine.AnalysisResult = await self.chess_engine.analysis(
            board=self.chess_board, limit=self.chess_engine_limits)
        try:
            return analysis, await analysis.wait()
        except asyncio.CancelledError:
            # Stop analysis
            analysis.stop()
            # Clear the current match data (saved for redrawing)
            self._current_match_data = {}
            raise

    def on_engine_analysis_finished(self, engine_analysis_future: asyncio.Future) -> None:
        # Check if the engine analysis was cancelled
        if engine_analysis_future.cancelled():
            return
        # Fetch the result (this will raise an exception if an exception was raised in the future)
        engine_analysis_result: AnalysisResultType = engine_analysis_future.result()

        # Keep track of the centipawn scores
        analysis, _ = engine_analysis_result
        player_score: chess.engine.Score = analysis.info["score"].pov(color=self.player_color)
        if isinstance(player_score, chess.engine.Cp):
            self._cp_scores.append((self.chess_board.ply(), player_score.score()))

        # Wait for the data to draw on the canvas
        event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self.draw_engine_analysis(analysis_result=engine_analysis_result), event_loop)

        # Send an unexpected message to the opponent
        if self.user_settings.troll_opponents:
            max_cp_score: int = max(self._cp_scores, default=(0, 0), key=lambda data: data[1])[1]
            if max_cp_score >= 150:
                asyncio.run_coroutine_threadsafe(self.troll_opponent(), event_loop)
                self.user_settings.troll_opponents = False

        # Perform a move if automatic moves are enabled
        if self.user_settings.auto_move and self.chess_board.turn == self.player_color:
            asyncio.run_coroutine_threadsafe(self.perform_move(analysis_result=engine_analysis_result), event_loop)

        # Execute engine control schemes
        self.execute_engine_control_schemes()

    async def troll_opponent(self) -> None:
        await self.owner_page.evaluate(expression="""
            lichess.pubsub.emit("socket.send", "talk", "Good game");
            setTimeout(() => lichess.pubsub.emit("socket.send", "talk", "Well played"), 500)
        """)

    def execute_engine_control_schemes(self) -> None:
        my_turn: bool = self.chess_board.turn == self.player_color
        move_number: int = self.chess_board.ply() // 2 + 1
        cp_score: int = 0 if not self._cp_scores else self._cp_scores[-1][1]
        max_cp_score: int = max(self._cp_scores, default=(0, 0), key=lambda data: data[1])[1]
        engine_depth: int = self.chess_engine_limits.depth

        for engine_control_scheme in self.engine_control_schemes:
            # Declare the locals
            locals_scope: dict = {
                "my_turn": my_turn,
                "move_number": move_number,
                "cp_score": cp_score,
                "max_cp_score": max_cp_score,
                "depth": engine_depth
            }
            # Execute the control scheme (note: this is unsafe, even without builtins)
            exec(f"if {engine_control_scheme.condition}: {engine_control_scheme.operation}", {"__builtins__": None},
                 locals_scope)

            # Update local variables
            new_depth: int = locals_scope.get("depth", engine_depth)
            if new_depth != engine_depth:
                self.chess_engine_limits.depth = new_depth

    async def on_move(self, move_data: dict) -> None:
        async with self._on_move_lock:
            uci_move: str = move_data["uci"]
            clock_data: dict = move_data.get("clock", {})
            promotion_data: dict = move_data.get("promotion", {})

            # Update the clock data
            if clock_data:
                self.chess_engine_limits.white_clock = clock_data["white"]
                self.chess_engine_limits.black_clock = clock_data["black"]

            # Manually fix promotions
            if promotion_data:
                piece_class_table: dict = {
                    "queen": "q",
                    "knight": "n",
                    "rook": "r",
                    "bishop": "b"
                }
                uci_move += piece_class_table[promotion_data["pieceClass"]]

            # Check if this is a valid move
            move: chess.Move = chess.Move.from_uci(uci=uci_move)
            if move in self.chess_board.legal_moves:
                # Cancel any pending engine analysis
                self.cancel_engine_analysis()
                # Clear the canvas
                await self.clear_canvas()
                # Push the move
                self.chess_board.push(move=move)

                # Queue engine analysis
                if not self.chess_board.is_game_over():
                    # Wait for the engine to be ready
                    await self.chess_engine.ping()
                    # Queue engine analysis
                    self.queue_engine_analysis()
            else:
                print(f"Attempted to perform an invalid move: {move=}, {move_data=}, {self.chess_board=}")

    def on_takeback_offer(self, origin: chess.Color) -> None:
        self._takeback_offer_origin = origin

    def on_takeback_cancelled(self) -> None:
        self._takeback_offer_origin = None

    async def on_takeback_accepted(self) -> None:
        # Determine the current turn
        current_turn: chess.Color = self.chess_board.turn

        # Undo moves based on the current turn and takeback offer origin
        if self._takeback_offer_origin == current_turn:
            self.chess_board.pop()
        # Handle an edge case for takebacks in computer games (takeback offer not sent)
        if self._takeback_offer_origin is None and current_turn == self.player_color:
            self.chess_board.pop()
        self.chess_board.pop()

        # Clear the takeback offer origin variable
        self._takeback_offer_origin = None

        # Update the centipawn scores (discard scores that aren't valid anymore)
        self._cp_scores = [score for score in self._cp_scores if score[0] < self.chess_board.ply()]

        # Clear the canvas
        await self.clear_canvas()
        # Queue engine analysis
        self.queue_engine_analysis()

    async def perform_move(self, analysis_result: AnalysisResultType) -> None:
        analysis, best_move = analysis_result
        assert best_move.move is not None
        best_move_uci: str = best_move.move.uci()

        piece_class_table: dict = {
            "q": "queen",
            "n": "knight",
            "r": "rook",
            "b": "bishop"
        }
        move_data: dict = {
            "origin": best_move_uci[:2],
            "destination": best_move_uci[2:4],
            "promotion": None if len(best_move_uci) < 5 else piece_class_table[best_move_uci[4]],
            "premove": self.user_settings.abuse_premoves
        }
        await self.owner_page.evaluate(
            expression="data => window.roundController.sendMove("
                       "data.origin, data.destination, data.promotion, {premove: data.premove});", arg=move_data)

    async def create_shadow_root(self) -> ElementHandle:
        # Create a shadow-root in the board element and add a resize observer
        board_locator: Locator = self.owner_page.locator(selector="cg-board")
        self._shadow_root = await board_locator.evaluate_handle(
            expression=self.scripts["create-shadow-root.js"], arg=self._canvas_height_offset)

        shadow_root_element: ElementHandle | None = self._shadow_root.as_element()
        assert shadow_root_element is not None
        return shadow_root_element

    async def try_rebuild_shadow_root(self) -> ElementHandle:
        # Check if the shadow root needs to be built
        if self._shadow_root is None:
            return await self.create_shadow_root()

        shadow_root_element: ElementHandle | None = self._shadow_root.as_element()
        assert shadow_root_element is not None

        # Check if the current shadow-root is connected to the DOM
        is_connected: bool = await shadow_root_element.evaluate(expression="shadowRoot => shadowRoot.isConnected;")
        if not is_connected:
            # Rebuild the shadow-root
            await self._shadow_root.dispose()
            self._shadow_root = None
            return await self.create_shadow_root()
        return shadow_root_element

    async def clear_canvas(self) -> None:
        # Cover an edge case when resize events are fired from the JS side
        if self._shadow_root is None:
            return

        shadow_root_element: ElementHandle | None = self._shadow_root.as_element()
        assert shadow_root_element is not None
        await shadow_root_element.eval_on_selector(
            selector="#drawing-canvas", expression="""canvas => {
                const context2d = canvas.getContext("2d");
                // Clear the canvas
                context2d.clearRect(0, 0, canvas.width, canvas.height);
            }""")

    async def get_board_data(self, shadow_root_element: ElementHandle) -> tuple[chess.Color, int, int, int]:
        # Determine the board orientation
        board_orientation: chess.Color = chess.WHITE if await shadow_root_element.evaluate(
            expression=self.scripts["get-board-orientation.js"]) == "orientation-white" else chess.BLACK
        # Determine the board (canvas) dimensions
        canvas_width, canvas_height = (await shadow_root_element.eval_on_selector(
            selector="#drawing-canvas",
            expression="canvas => ({width: canvas.width, height: canvas.height});")).values()
        piece_size: int = canvas_width // 8
        # Return the results
        return board_orientation, canvas_width, canvas_height, piece_size

    async def draw_engine_analysis(self, analysis_result: AnalysisResultType) -> None:
        analysis, best_move = analysis_result
        assert best_move.move is not None

        best_move_uci: str = best_move.move.uci()
        ponder_move_uci: str = best_move.ponder.uci() if best_move.ponder is not None else ""

        analysis_score: chess.engine.PovScore = analysis.info["score"]
        player_score: chess.engine.Score = analysis_score.pov(color=self.player_color)
        score: str = f"{player_score.score()}" if not player_score.is_mate() else f"#{player_score.mate()}"

        # Determine the background color to use for the score
        score_color: str = "#BDC3C7"
        if isinstance(player_score, chess.engine.Mate):
            score_color = "#2ECC71" if player_score.mate() >= 1 else "#E74C3C"
        elif isinstance(player_score, chess.engine.Cp) and player_score.score() != 0:
            score_color = "#2ECC71" if player_score.score() > 0 else "#E74C3C"

        # Rebuild the shadow root if it's required
        shadow_root_element: ElementHandle = await self.try_rebuild_shadow_root()

        # Fetch board data
        board_orientation, canvas_width, canvas_height, piece_size = await self.get_board_data(
            shadow_root_element=shadow_root_element)

        # Draw on the canvas
        match_data: dict = {
            "uiOffsets": {
                "x": 0,
                "y": canvas_height - self._canvas_height_offset
            },
            "score": score,
            "scoreColor": score_color,
            "depth": self.chess_engine_limits.depth,
            "bestMove": {
                "uci": best_move_uci,
                "coordinates": self.calculate_move_positions(board_orientation, piece_size, best_move_uci),
                "color": "#2ECC71"
            },
            "ponderMove": {
                "uci": ponder_move_uci,
                "coordinates": self.calculate_move_positions(board_orientation, piece_size, ponder_move_uci),
                "color": "#2980B9"
            } if best_move.ponder else None,
            "pv": [move.uci() for move in analysis.info["pv"]]
        }
        self._current_match_data = match_data
        await shadow_root_element.eval_on_selector(
            selector="#drawing-canvas", expression=self.scripts["draw-data.js"], arg=match_data)

    async def redraw_existing_engine_analysis(self, new_board_width: int) -> None:
        if not self._current_match_data:
            return
        # Avoid attempting to create duplicate shadow roots
        if self._shadow_root is None:
            return

        shadow_root_element: ElementHandle = await self.try_rebuild_shadow_root()
        # Fetch board data
        board_orientation, canvas_width, canvas_height, piece_size = await self.get_board_data(
            shadow_root_element=shadow_root_element)
        # Handle an edge case where the new piece size is 0 after the board is flipped (observer callback)
        if new_board_width:
            assert new_board_width == canvas_width

        # Recalculate the move positions
        self._current_match_data["bestMove"]["coordinates"] = self.calculate_move_positions(
            board_orientation=board_orientation, piece_size=piece_size,
            uci_move=self._current_match_data["bestMove"]["uci"])
        if self._current_match_data["ponderMove"]:
            self._current_match_data["ponderMove"]["coordinates"] = self.calculate_move_positions(
                board_orientation=board_orientation, piece_size=piece_size,
                uci_move=self._current_match_data["ponderMove"]["uci"])

        # Redraw the data
        await shadow_root_element.eval_on_selector(
            selector="#drawing-canvas", expression=self.scripts["draw-data.js"], arg=self._current_match_data)

    @staticmethod
    def calculate_move_positions(board_orientation: chess.Color,
                                 piece_size: int, uci_move: str) -> dict[str, float]:
        canvas_width = canvas_height = piece_size * 8

        # Calculate the position data
        from_x: float = (ord(uci_move[0]) - ord("a")) * piece_size + piece_size / 2
        from_y: float = canvas_height - (ord(uci_move[1]) - ord("1") + 1) * piece_size + piece_size / 2
        to_x: float = (ord(uci_move[2]) - ord("a")) * piece_size + piece_size / 2
        to_y: float = canvas_height - (ord(uci_move[3]) - ord("1") + 1) * piece_size + piece_size / 2

        # Handle board inversion
        if board_orientation == chess.BLACK:
            from_x = canvas_width - from_x
            from_y = canvas_height - from_y
            to_x = canvas_width - to_x
            to_y = canvas_height - to_y
        # Return the position data
        return {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}

    # Note: do not call self._shadow_root.dispose(), it'll prevent Playwright from closing
    async def shutdown(self) -> None:
        # Cancel the engine analysis task
        self.cancel_engine_analysis()
        # Stop the engine process
        await self.chess_engine.quit()
        self.transport.close()
        await asyncio.sleep(0)


@dataclass
class Lichess(BrowserHandler):
    chess_rounds: dict[str, Round] = field(default_factory=dict)
    _round_cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.browser_context.on(event="page", f=self.on_page)
        self.browser_context.on(event="close", f=self.on_context_close)

    # https://github.com/python/mypy/issues/12909
    @staticmethod
    @asynccontextmanager
    async def create(playwright: Playwright) -> AsyncIterator[Lichess]:  # type: ignore[override]
        browser_instance: Browser = await playwright.chromium.launch(headless=False)
        browser_context: BrowserContext = await browser_instance.new_context()
        lichess_handler: Lichess = Lichess(browser=browser_instance, browser_context=browser_context)

        try:
            first_page: Page = await browser_context.new_page()
            await first_page.goto(url="https://lichess.org", wait_until="networkidle")
            yield lichess_handler
        finally:
            await lichess_handler.close()

    async def wait(self) -> None:
        await self.browser_context.pages[0].pause()

    async def close(self) -> None:
        await self.browser_context.close()
        await self.browser.close()

    async def on_page(self, page: Page) -> None:
        # Expose engine bindings
        await page.expose_binding("set_depth", lambda source, depth: self.set_depth(source["page"], depth))
        await page.expose_binding("toggle_automove", lambda source: self.toggle_automove(source["page"]))
        # Expose canvas bindings
        await page.expose_binding("redraw_existing_engine_analysis",
                                  lambda source, piece_size: self.redraw_existing_engine_analysis(
                                      source["page"], piece_size))
        # Register an event listener for websocket events
        page.on("websocket", functools.partial(self.on_websocket_created, page=page))

        # Modify the round setup javascript to be able to access the internals
        async def handle_round_route(route: Route) -> None:
            response: APIResponse = await route.fetch()
            response_body: str = await response.text()

            new_code: str = "window.roundController = this;"

            round_constructor_string: str = "var ft=class{constructor(t,o){"
            round_constructor_index: int = response_body.find(round_constructor_string) + len(round_constructor_string)
            response_body = response_body[:round_constructor_index] + new_code + response_body[round_constructor_index:]
            await route.fulfill(response=response, body=response_body)
        await page.route("**/round.min.js", handle_round_route)

    async def on_context_close(self, _: BrowserContext) -> None:
        for game_round in self.chess_rounds.values():
            await game_round.shutdown()
        self.chess_rounds.clear()

    @staticmethod
    def generate_round_identifier(web_socket: WebSocket) -> str:
        # Generate a pattern to match the WS URL
        url_pattern: Pattern[str] = re.compile(
            r"wss://socket\d\.lichess\.org/play/"
            r"(?P<round_identifier>\w{12})/"
            r"v(?P<socket_version>\d+)\?sri=\w{12}&v=(?P<ply_number>\d+)"
        )
        # Attempt to match the URL pattern
        if not (match := re.match(pattern=url_pattern, string=web_socket.url)):
            raise ValueError(f"{web_socket.url} failed to match the url pattern.")
        # Return the round identifier
        return match.group("round_identifier")

    def find_round(self, page: Page) -> Round:
        page_round_identifier: str = page.url[page.url.rfind("/") + 1:]
        for round_identifier, chess_round in self.chess_rounds.items():
            if round_identifier == page_round_identifier or round_identifier.startswith(page_round_identifier):
                return chess_round
        raise ValueError(f"{page_round_identifier} does not have an associated Round instance.")

    async def on_websocket_created(self, web_socket: WebSocket, page: Page) -> None:
        try:
            round_identifier: str = self.generate_round_identifier(web_socket=web_socket)
        except ValueError:
            return

        # Post a notification
        print(f"Started a game: {round_identifier}")

        # Locate the initial match data
        game_data: dict = {}
        for script in await page.locator(r"//script").all():
            script_data: str = await script.inner_text()
            if not script_data.startswith("lichess.load.then"):
                continue

            game_data = json.loads(
                script_data[script_data.find("{\"data\":"):script_data.rfind(")})")])
            game_data = game_data.get("data", {})
            break
        if not game_data:
            print("Failed to locate the initial game data.")
            return

        # Add a new round instance
        player_color: chess.Color = chess.WHITE if game_data["player"]["color"] == "white" else chess.BLACK
        time_increment: float = float(game_data["clock"]["increment"]) if "clock" in game_data else 0.0
        move_data: list[dict] = game_data.get("steps", game_data.get("treeParts", []))
        self.chess_rounds[round_identifier] = await Round.create(
            page=page, identifier=round_identifier, player_color=player_color,
            move_data=move_data, time_increment=time_increment)

        # Register web socket events
        web_socket.on("framereceived", functools.partial(
            self.on_websocket_message, chess_round=self.chess_rounds[round_identifier], from_client=False))
        web_socket.on("framesent", functools.partial(
            self.on_websocket_message, chess_round=self.chess_rounds[round_identifier], from_client=True))

        # Remove the game round from the internal list
        web_socket.on("close", self.on_websocket_closed)

    async def on_websocket_message(self, payload: str, chess_round: Round, from_client: bool) -> None:
        # Ignore messages from the client
        if from_client:
            return

        # Parse the message
        try:
            parsed_payload: Any = json.loads(payload)
            if not isinstance(parsed_payload, dict):
                return
        except json.decoder.JSONDecodeError:
            return

        # Parse the message
        message_type: str = parsed_payload.get("t", "unknown")
        if message_type == "move":
            move_data: dict = parsed_payload["d"]
            await chess_round.on_move(move_data=move_data)
        elif message_type == "takebackOffers":
            takeback_data: dict | None = parsed_payload["d"]
            if not takeback_data:
                chess_round.on_takeback_cancelled()
            else:
                turn_origin: chess.Color = chess.WHITE if next(iter(takeback_data)) == "white" else chess.BLACK
                chess_round.on_takeback_offer(origin=turn_origin)
        elif message_type == "reload":
            reload_data: dict | None = parsed_payload["d"]
            if reload_data is None:  # avoid an edge case for when a rematch is accepted
                await chess_round.on_takeback_accepted()
        elif message_type == "endData":
            # Cancel any pending engine analysis
            chess_round.cancel_engine_analysis()
            # Clear the canvas
            await chess_round.clear_canvas()
            # Shutdown the chess round instance
            await self.perform_round_cleanup(round_identifier=chess_round.identifier)

    def set_depth(self, page: Page, depth: int) -> str:
        try:
            # Attempt to find the associated chess round
            chess_round: Round = self.find_round(page=page)
            # Update the engine depth
            chess_round.update_settings(settings={"engine-depth": depth})
            chess_round.queue_engine_analysis()
            return f"set engine depth to {depth}"
        except ValueError:
            return "could not find a chess round"

    def toggle_automove(self, page: Page) -> str:
        try:
            # Attempt to find the associated chess round
            chess_round: Round = self.find_round(page=page)
            # Toggle automatic moves
            new_value: bool = not chess_round.user_settings.auto_move
            chess_round.update_settings(settings={"auto-move": new_value})
            return f"{'enabled' if new_value else 'disabled'} automatic moves"
        except ValueError:
            return "could not find a chess round"

    async def redraw_existing_engine_analysis(self, page: Page, new_board_width: int) -> None:
        round_identifier: str = page.url[page.url.rfind("/") + 1:]
        for web_socket_url, chess_round in self.chess_rounds.items():
            if round_identifier not in web_socket_url:
                continue
            await chess_round.redraw_existing_engine_analysis(new_board_width=new_board_width)
            return

    async def perform_round_cleanup(self, round_identifier: str) -> None:
        async with self._round_cleanup_lock:
            # Check if the game has already ended
            if round_identifier not in self.chess_rounds:
                return

            # Push a notification
            print(f"Ended a game: {round_identifier}")

            # Perform cleanup
            await self.chess_rounds[round_identifier].shutdown()
            del self.chess_rounds[round_identifier]

    async def on_websocket_closed(self, web_socket: WebSocket) -> None:
        await self.perform_round_cleanup(round_identifier=self.generate_round_identifier(web_socket=web_socket))


async def main() -> int:
    async with async_playwright() as playwright:
        async with Lichess.create(playwright=playwright) as lichess_handler:
            await lichess_handler.wait()

    # Execute pending tasks (avoid setting chess.engine.EventLoopPolicy)
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(), debug=True))
