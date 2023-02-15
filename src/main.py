from __future__ import annotations

import asyncio
import functools
import json
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Pattern, TypeVar

import chess
from chess.engine import Limit as ChessEngineLimit, UciProtocol as ChessEngine, popen_uci as open_chess_engine
from playwright.async_api import Browser, BrowserContext, ElementHandle, JSHandle, Locator, Page, Playwright, \
    WebSocket, async_playwright

BINARIES_PATH: Path = Path.cwd() / "binaries"
STOCKFISH_PATH: Path = BINARIES_PATH / "stockfish"
STOCKFISH_ENGINE_PATH: Path = STOCKFISH_PATH / "stockfish-windows-2022-x86-64-avx2.exe"
STOCKFISH_SETTINGS_PATH: Path = STOCKFISH_PATH / "settings.json"


BrowserHandlerType = TypeVar("BrowserHandlerType", bound="BrowserHandler")


@dataclass
class BrowserHandler(ABC):
    browser: Browser
    browser_context: BrowserContext

    @staticmethod
    @abstractmethod
    async def create(playwright: Playwright) -> AsyncIterator[BrowserHandlerType]:
        raise NotImplementedError


@dataclass
class Round:
    owner_page: Page
    player_color: chess.Color
    chess_board: chess.Board
    transport: asyncio.SubprocessTransport
    chess_engine: ChessEngine
    chess_engine_limits: ChessEngineLimit = field(default_factory=ChessEngineLimit)
    _chess_engine_analysis_task: asyncio.Task | None = None
    _takeback_offer_origin: str = ""
    _shadow_root: JSHandle | None = None

    @staticmethod
    async def create(page: Page, player_color: str, move_data: list[dict]) -> Round:
        # Setup the chess board
        chess_board: chess.Board = chess.Board(fen=move_data[0]["fen"])
        for move in move_data[1:]:
            chess_board.push_uci(move["uci"])

        # Read the preferred settings from disk
        settings: dict = json.loads(STOCKFISH_SETTINGS_PATH.read_text())

        # Spawn an engine instance and configure the engine
        transport, engine = await open_chess_engine(str(STOCKFISH_ENGINE_PATH))
        await engine.configure(options=settings["uci-settings"])

        # Create a round instance
        round_instance: Round = Round(owner_page=page,
                                      player_color=chess.WHITE if player_color == "white" else chess.BLACK,
                                      chess_board=chess_board, transport=transport, chess_engine=engine)

        # Configure the engine limits
        round_instance.chess_engine_limits.depth = settings["engine-limits"]["depth"]

        # Create the shadow dom
        await round_instance.create_shadow_root()

        # If we are to make the first move then queue up engine analysis
        if player_color == "white":
            round_instance.queue_engine_analysis()

        # Return the round instance
        return round_instance

    def queue_engine_analysis(self) -> None:
        # Cancel any pending tasks
        if self._chess_engine_analysis_task is not None and not self._chess_engine_analysis_task.done():
            self._chess_engine_analysis_task.cancel(msg="New engine analysis task queued.")

        # Check if the engine is ready
        self._chess_engine_analysis_task = asyncio.create_task(self.find_best_move(), name="Best move task")
        self._chess_engine_analysis_task.add_done_callback(self.on_engine_analysis_finished)

    async def find_best_move(self) -> tuple[chess.engine.AnalysisResult, chess.engine.BestMove] | tuple[None, None]:
        analysis: chess.engine.AnalysisResult = await self.chess_engine.analysis(
            board=self.chess_board, limit=self.chess_engine_limits)
        try:
            return analysis, await analysis.wait()
        except asyncio.CancelledError:
            analysis.stop()
            return None, None

    def on_engine_analysis_finished(self, _: asyncio.Future) -> None:
        # Redraw the canvas
        asyncio.create_task(self.redraw_canvas())

    async def on_move(self, move_data: dict) -> None:
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
                "knight": "k",
                "rook": "r",
                "bishop": "b"
            }
            uci_move += piece_class_table[promotion_data["pieceClass"]]

        # Check if this is a valid move
        move: chess.Move = chess.Move.from_uci(uci=uci_move)
        if move in self.chess_board.legal_moves:
            # Push the move
            self.chess_board.push(move=move)
            # Clear the canvas
            await self.clear_canvas()

            # Queue engine analysis
            if not self.chess_board.is_game_over():
                self.queue_engine_analysis()
        else:
            print(f"Attempted to perform an invalid move: {move=}, {move_data=}")

    def on_takeback_offer(self, origin: str) -> None:
        self._takeback_offer_origin = origin

    async def on_takeback_accepted(self) -> None:
        # Determine the current turn
        current_turn: str = "white" if self.chess_board.turn == chess.WHITE else "black"

        # Undo moves based on the current turn and takeback offer origin
        # Handle an edge case for takebacks in computer games (takeback offer not sent)
        if not self._takeback_offer_origin or current_turn == self._takeback_offer_origin:
            self.chess_board.pop()
        self.chess_board.pop()

        # Clear the takeback offer origin variable
        self._takeback_offer_origin = ""

        # Clear the canvas
        await self.clear_canvas()
        # Queue engine analysis
        self.queue_engine_analysis()

    async def create_shadow_root(self) -> None:
        # Create a shadow-root in the board element and add a resize observer
        board_locator: Locator = self.owner_page.locator(selector="cg-board")
        self._shadow_root = await board_locator.evaluate_handle(
            expression="""boardElement => {
                // Create a closed shadow-root
                const shadowRoot = boardElement.attachShadow({mode: "closed"});
                shadowRoot.innerHTML = `
                    <canvas id="drawing-canvas" style="
                        position: relative;
                        z-index: 3;
                        pointer-events: none;"></canvas>
                    <slot></slot>`;

                // Fetch the canvas element
                const canvasElement = shadowRoot.querySelector("canvas");
                canvasElement.width = boardElement.clientWidth;
                canvasElement.height = boardElement.clientHeight;

                // Attach a resize observer to the board element
                new ResizeObserver(entries => entries.forEach(entry => {
                    // Set the new size
                    const contentRect = entry.contentRect;
                    canvasElement.width = contentRect.width;
                    canvasElement.height = contentRect.height;

                    // Call the redraw_canvas binding
                    window.redraw_canvas();
                })).observe(boardElement);
                // Return the shadow-root for later access
                return shadowRoot;
            }""")

    async def try_rebuild_shadow_root(self) -> None:
        if self._shadow_root is None:
            await self.create_shadow_root()
            return

        shadow_root_element: ElementHandle | None = self._shadow_root.as_element()
        assert shadow_root_element is not None

        # Check if the shadow-root is connected to the DOM
        is_connected: bool = await shadow_root_element.evaluate(expression="shadowRoot => shadowRoot.isConnected;")
        if not is_connected:
            # Rebuild the shadow-root
            await self._shadow_root.dispose()
            self._shadow_root = None
            await self.create_shadow_root()

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

    async def redraw_canvas(self) -> None:
        # Cover an edge case when resize events are fired from the JS side
        if self._shadow_root is None:
            return

        # Rebuild the shadow root if it's required
        await self.try_rebuild_shadow_root()

        shadow_root_element: ElementHandle | None = self._shadow_root.as_element()
        assert shadow_root_element is not None

        # Check if the engine has calculated the best move
        if self._chess_engine_analysis_task is None:
            return
        if not self._chess_engine_analysis_task.done():
            return

        # Fetch the results
        # TODO: race condition possible here (TypeError: cannot unpack non-iterable NoneType object)
        # TODO: assuming that a move was sent while in the function, so the result was changed (use a lock?)
        analysis, best_move = self._chess_engine_analysis_task.result()
        if best_move is None:
            return
        assert best_move.move is not None

        best_move_uci: str = best_move.move.uci()
        ponder_move_uci: str = best_move.ponder.uci() if best_move.ponder is not None else ""

        analysis_score: chess.engine.PovScore = analysis.info["score"]
        player_score: chess.engine.Score = analysis_score.pov(color=self.player_color)
        score: str = f"{player_score.score()}" if player_score.mate() is None else f"#{player_score.mate()}"

        # Determine the background color to use for the score
        score_color: str
        if player_score.is_mate():
            score_color = "#2ECC71" if player_score.mate() >= 1 else "#E74C3C"
        elif player_score.score() != 0:
            score_color = "#2ECC71" if player_score.score() > 0 else "#E74C3C"
        else:
            score_color = "#BDC3C7"

        # Calculate the move positions
        board_orientation: str = await shadow_root_element.evaluate(
            expression="""shadowRoot => {
                const container = shadowRoot.host.parentNode.parentNode;
                return Array.from(container.classList).find(className => className.includes("orientation"));
            }""")
        canvas_width, canvas_height = (await shadow_root_element.eval_on_selector(
            selector="#drawing-canvas",
            expression="canvas => ({width: canvas.width, height: canvas.height});")).values()
        assert canvas_width == canvas_height

        # Calculate move positions
        best_move_position_data: dict[str, float] = self.calculate_move_positions(
            board_orientation=board_orientation, piece_size=canvas_width // 8, uci_move=best_move_uci)
        ponder_move_position_data: dict[str, float] = self.calculate_move_positions(
            board_orientation=board_orientation, piece_size=canvas_width // 8, uci_move=ponder_move_uci)

        # Draw on the canvas
        await shadow_root_element.eval_on_selector(
            selector="#drawing-canvas", expression=f"""canvas => {{
                const context2d = canvas.getContext("2d");
                // Clear the canvas
                context2d.clearRect(0, 0, canvas.width, canvas.height);

                const previousCompositeOperation = context2d.globalCompositeOperation;
                const drawArrow = (lineWidth, color, alpha, position) => {{
                    const deltaX = position.to_x - position.from_x;
                    const deltaY = position.to_y - position.from_y;
                    const angle = Math.atan2(deltaY, deltaX);
                    const headLength = 10;

                    context2d.save();
                    context2d.lineCap = "round";
                    context2d.lineWidth = lineWidth;
                    context2d.strokeStyle = color;
                    context2d.globalAlpha = alpha;

                    // Draw the arrow
                    context2d.beginPath();
                    context2d.moveTo(position.from_x, position.from_y);
                    context2d.lineTo(position.to_x, position.to_y);
                    context2d.moveTo(position.to_x, position.to_y);
                    context2d.lineTo(position.to_x - headLength * Math.cos(angle - Math.PI / 6),
                                     position.to_y - headLength * Math.sin(angle - Math.PI / 6));
                    context2d.moveTo(position.to_x, position.to_y);
                    context2d.lineTo(position.to_x - headLength * Math.cos(angle + Math.PI / 6),
                                     position.to_y - headLength * Math.sin(angle + Math.PI / 6));
                    context2d.stroke();
                    context2d.closePath();
                    context2d.restore();
                }};

                // Render the ponder move
                if ("{ponder_move_uci}" !== "")
                    drawArrow(4, "#2980B9", 0.75, {ponder_move_position_data});
                // Render the best move
                drawArrow(4, "#2ECC71", 0.75, {best_move_position_data});

                // Compute the text array
                let textArray = [];
                textArray.push("best: {best_move_uci}");
                if ("{ponder_move_uci}" !== "")
                    textArray.push("ponder: {ponder_move_uci}");
                textArray.push("score: {score}");
                textArray.push("depth: {self.chess_engine_limits.depth}");

                // Calculate the bounding box
                let maxWidth = 0, maxHeight = 0;
                for (const text of textArray) {{
                    const textSize = context2d.measureText(text);
                    if (textSize.width > maxWidth)
                        maxWidth = textSize.width;
                    maxHeight += 10 + 1;
                }}

                // Render the text
                context2d.globalAlpha = 1;
                context2d.fillStyle = "#ffffff";
                context2d.fillRect(1, 1, maxWidth + 4, maxHeight);
                context2d.fillStyle = "#000000";
                for (const [index, text] of textArray.entries()) {{
                    if (text.includes("score")) {{
                        context2d.fillStyle = "{score_color}"
                        context2d.fillRect(1, index * 10 + 2, maxWidth + 4, 10);
                        context2d.fillStyle = "#000000";
                    }}
                    context2d.fillText(text, 3, (index + 1) * 10);
                }}
            }}""")

    @staticmethod
    def calculate_move_positions(board_orientation: str,
                                 piece_size: int, uci_move: str) -> dict[str, float]:
        if not uci_move:
            return dict()
        canvas_width = canvas_height = piece_size * 8

        # Calculate the position data
        from_x: float = (ord(uci_move[0]) - ord("a")) * piece_size + piece_size / 2
        from_y: float = canvas_height - (ord(uci_move[1]) - ord("1") + 1) * piece_size + piece_size / 2
        to_x: float = (ord(uci_move[2]) - ord("a")) * piece_size + piece_size / 2
        to_y: float = canvas_height - (ord(uci_move[3]) - ord("1") + 1) * piece_size + piece_size / 2

        # Handle board inversion
        if board_orientation != "orientation-white":
            from_x = canvas_width - from_x
            from_y = canvas_height - from_y
            to_x = canvas_width - to_x
            to_y = canvas_height - to_y
        # Return the position data
        return {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}

    # Note: do not call self._shadow_root.dispose(), it'll prevent Playwright from closing
    async def shutdown(self) -> None:
        # Cancel the engine analysis task
        if self._chess_engine_analysis_task is not None and not self._chess_engine_analysis_task.done():
            self._chess_engine_analysis_task.cancel(msg="Shutting down.")
        # Stop the engine process
        await self.chess_engine.quit()
        self.transport.close()
        await asyncio.sleep(0)


@dataclass
class Lichess(BrowserHandler):
    chess_rounds: dict[str, Round] = field(default_factory=dict)

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
        # https://github.com/microsoft/playwright-python/issues/1748
        await asyncio.wait_for(self.browser_context._impl_obj._pause(), None)

    async def close(self) -> None:
        await self.browser_context.close()
        await self.browser.close()

    async def on_page(self, page: Page) -> None:
        # Expose engine bindings
        await page.expose_binding("get_best_move", lambda source, depth=0: self.get_best_move(source["page"], depth))
        await page.expose_binding("set_depth", lambda source, depth: self.set_depth(source["page"], depth))
        await page.expose_binding("display_board", lambda source: self.display_board(source["page"]))
        # Expose canvas bindings
        await page.expose_binding("redraw_canvas", lambda source: self.redraw_canvas(source["page"]))
        # Register an event listener for websocket events
        page.on("websocket", functools.partial(self.on_websocket_created, page=page))
        # TODO: shutdown round instances on reload/redirect

    async def on_context_close(self, _: BrowserContext) -> None:
        for game_round in self.chess_rounds.values():
            await game_round.shutdown()
        self.chess_rounds.clear()

    async def on_websocket_created(self, web_socket: WebSocket, page: Page) -> None:
        # Check to make sure the web socket url matches the filter
        url_pattern: Pattern[str] = re.compile(
            r"wss://socket\d\.lichess\.org/"
            r"(?:play/\w{12}|watch/\w{8}/(?:white|black))/"
            r"v(?P<socket_version>\d)\?sri=\w{12}(?:&v=(?P<move_number>\d+))?")
        match: re.Match | None = re.match(pattern=url_pattern, string=web_socket.url)
        if match is None:
            return

        # Post a notification
        print(f"Started a game from: {web_socket.url}")

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
        player_color: str = game_data["player"]["color"]
        move_data: list[dict] = game_data.get("steps", game_data.get("treeParts", []))
        self.chess_rounds[web_socket.url] = await Round.create(
            page=page, player_color=player_color, move_data=move_data)

        # Register web socket events
        web_socket.on("framereceived", functools.partial(self.on_websocket_message,
                                                         socket_identifier=web_socket.url,
                                                         chess_round=self.chess_rounds[web_socket.url],
                                                         from_client=False))
        web_socket.on("framesent", functools.partial(self.on_websocket_message,
                                                     socket_identifier=web_socket.url,
                                                     chess_round=self.chess_rounds[web_socket.url],
                                                     from_client=True))

        # Remove the game round from the internal list
        web_socket.on("close", self.on_websocket_closed)

    async def on_websocket_message(self, payload: str,
                                   socket_identifier: str, chess_round: Round, from_client: bool) -> None:
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
            takeback_data: dict = parsed_payload["d"]
            chess_round.on_takeback_offer(origin=next(iter(takeback_data)))
        elif message_type == "reload":
            reload_data: dict | None = parsed_payload["d"]
            if reload_data is None:  # avoid an edge case for when a rematch is accepted
                await chess_round.on_takeback_accepted()
        elif message_type == "endData":
            # Clear the canvas
            await chess_round.clear_canvas()
            # Shutdown the chess round instance
            await self.perform_round_cleanup(socket_identifier=socket_identifier)

    async def get_best_move(self, page: Page, depth: int) -> str:
        # Find the correct socket
        round_identifier: str = page.url[page.url.rfind("/") + 1:]
        for web_socket_url, chess_round in self.chess_rounds.items():
            if round_identifier not in web_socket_url:
                continue

            # Set the depth
            assert chess_round.chess_engine_limits.depth is not None
            previous_depth: int = chess_round.chess_engine_limits.depth
            if depth and depth != previous_depth:
                assert depth > 0
                chess_round.chess_engine_limits.depth = depth

            # Calculate the best move
            analysis, best_move = await chess_round.find_best_move()
            # Revert the depth
            if depth and depth != previous_depth:
                chess_round.chess_engine_limits.depth = previous_depth

            assert analysis is not None
            assert best_move is not None
            assert best_move.move is not None

            # Revert the depth
            if depth and depth != previous_depth:
                chess_round.chess_engine_limits.depth = previous_depth

            # Return the result
            ponder_move_info: str = f" ponder move={best_move.ponder.uci()}" if best_move.ponder is not None else ""
            score: chess.engine.PovScore = analysis.info["score"]
            return f"best move={best_move.move.uci()}{ponder_move_info} score={score}"
        return "couldn't find a game"

    def set_depth(self, page: Page, depth: int) -> None:
        # Find the correct socket
        round_identifier: str = page.url[page.url.rfind("/") + 1:]
        for web_socket_url, chess_round in self.chess_rounds.items():
            if round_identifier not in web_socket_url:
                continue
            # Set the depth
            chess_round.chess_engine_limits.depth = depth
            return

    def display_board(self, page: Page) -> str:
        # Find the correct socket
        round_identifier: str = page.url[page.url.rfind("/") + 1:]
        for web_socket_url, chess_round in self.chess_rounds.items():
            if round_identifier not in web_socket_url:
                continue
            # Set the depth
            return str(chess_round.chess_board)
        return ""

    async def redraw_canvas(self, page: Page) -> None:
        round_identifier: str = page.url[page.url.rfind("/") + 1:]
        for web_socket_url, chess_round in self.chess_rounds.items():
            if round_identifier not in web_socket_url:
                continue
            await chess_round.redraw_canvas()
            return

    async def perform_round_cleanup(self, socket_identifier: str) -> None:
        # Check if the game has already ended
        if socket_identifier not in self.chess_rounds:
            return

        # Push a notification
        print(f"Shutting down a game: {socket_identifier}")

        # Perform cleanup
        await self.chess_rounds[socket_identifier].shutdown()
        del self.chess_rounds[socket_identifier]

    async def on_websocket_closed(self, web_socket: WebSocket) -> None:
        await self.perform_round_cleanup(socket_identifier=web_socket.url)


async def main() -> int:
    async with async_playwright() as playwright:
        async with Lichess.create(playwright=playwright) as lichess_handler:
            await lichess_handler.wait()

    # Execute pending tasks (avoid setting chess.engine.EventLoopPolicy)
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
