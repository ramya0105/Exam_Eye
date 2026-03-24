document.addEventListener("DOMContentLoaded", () => {
    const toggles = document.querySelectorAll("[data-toggle-password]");

    toggles.forEach((button) => {
        button.addEventListener("click", () => {
            const inputId = button.getAttribute("data-toggle-password");
            const input = document.getElementById(inputId);
            if (!input) return;

            const isPassword = input.type === "password";
            input.type = isPassword ? "text" : "password";
            button.innerHTML = isPassword
                ? '<i class="fas fa-eye-slash"></i>'
                : '<i class="fas fa-eye"></i>';
        });
    });
});
