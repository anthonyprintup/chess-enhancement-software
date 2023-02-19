// noinspection BadExpressionStatementJS
boardElement => {
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
    // Set its dimensions
    canvasElement.width = boardElement.clientWidth;
    canvasElement.height = boardElement.clientHeight;

    // Attach a resize observer to the board element
    new ResizeObserver(entries => entries.forEach(entry => {
        // Set the new size
        const contentRect = entry.contentRect;
        canvasElement.width = contentRect.width;
        canvasElement.height = contentRect.height;

        // Redraw any existing engine analysis (argument 1 is the new size of an individual piece)
        // noinspection JSUnresolvedFunction
        window.redraw_existing_engine_analysis(canvasElement.width / 8);
    })).observe(boardElement);
    // Return the shadow-root for later access
    return shadowRoot;
};
