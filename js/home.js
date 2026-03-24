document.addEventListener("DOMContentLoaded", () => {
    const tabs = document.querySelectorAll(".tab-btn");
    const panels = document.querySelectorAll(".tab-panel");

    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");
            const targetPanel = document.getElementById(targetId);
            if (!targetPanel) return;

            tabs.forEach((btn) => {
                btn.classList.remove("active");
                btn.setAttribute("aria-selected", "false");
            });

            panels.forEach((panel) => {
                panel.classList.remove("active");
                panel.hidden = true;
            });

            tab.classList.add("active");
            tab.setAttribute("aria-selected", "true");
            targetPanel.hidden = false;
            targetPanel.classList.add("active");
        });
    });
});
