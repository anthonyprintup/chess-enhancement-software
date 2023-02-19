// noinspection BadExpressionStatementJS
shadowRoot => {
    const container = shadowRoot.host.parentNode.parentNode;
    return Array.from(container.classList).find(className => className.includes("orientation"));
}
