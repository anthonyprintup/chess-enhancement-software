// noinspection JSUnresolvedVariable
// noinspection BadExpressionStatementJS
(canvas, matchData) => {
    const context2d = canvas.getContext("2d");
    // Clear the canvas
    context2d.clearRect(0, 0, canvas.width, canvas.height);

    const drawArrow = (lineWidth, color, alpha, position) => {
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
    };

    // Render the ponder move
    if (matchData.ponderMove)
        drawArrow(4, matchData.ponderMove.color, 0.75, matchData.ponderMove.coordinates);
    // Render the best move
    drawArrow(4, matchData.bestMove.color, 0.75, matchData.bestMove.coordinates);

    // Compute the text array
    let textArray = [];
    let maxWidth = 0;
    const pushText = (label, text, backgroundColor = "#ffffff") => {
        // Format the text
        const formattedText = `${label}: ${text}`;

        // Measure the text
        const textMetrics = context2d.measureText(formattedText);
        if (textMetrics.width > maxWidth)
            maxWidth = textMetrics.width;
        const textHeight = textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent;

        // Push the text into the array
        let yOffset = 0;
        if (textArray.length) {
            const lastElement = textArray.at(-1);
            yOffset = lastElement.yOffset + lastElement.height;
        }
        textArray.push({
            text: formattedText,
            backgroundColor: backgroundColor,
            yOffset: yOffset,
            height: textHeight
        });
    };

    pushText("best", matchData.bestMove.uci);
    if (matchData.ponderMove)
        pushText("ponder", matchData.ponderMove.uci);
    pushText("score", matchData.score, matchData.scoreColor);
    pushText("depth", matchData.depth);

    // Render the text
    context2d.globalAlpha = 1;
    // noinspection JSUnusedLocalSymbols
    for (const {_, backgroundColor, yOffset, height} of textArray) {
        context2d.fillStyle = backgroundColor;
        context2d.fillRect(matchData.uiOffsets.x, matchData.uiOffsets.y + yOffset,maxWidth + 4, height);
    }
    // noinspection JSUnusedLocalSymbols
    for (const {text, _, yOffset, height} of textArray) {
        context2d.fillStyle = "#000000";
        context2d.fillText(text, matchData.uiOffsets.x + 2, matchData.uiOffsets.y + yOffset + height - 2);
    }
}
