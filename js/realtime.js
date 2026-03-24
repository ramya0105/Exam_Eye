const ids = {
    duration: document.getElementById("duration-sec"),
    frames: document.getElementById("total-frames"),
    away: document.getElementById("looking-away"),
    eyes: document.getElementById("eyes-closed"),
    phone: document.getElementById("phone-detected"),
    people: document.getElementById("unauth-people"),
    status: document.getElementById("live-status"),
    feed: document.getElementById("realtime-feed"),
    stopBtn: document.getElementById("stop-btn"),
    restartBtn: document.getElementById("restart-btn"),
};

async function refreshScores() {
    try {
        const res = await fetch("/realtime/scores", { cache: "no-store" });
        if (!res.ok) return;

        const data = await res.json();
        ids.duration.textContent = `${data.duration_sec}s`;
        ids.frames.textContent = data.total_frames;
        ids.away.textContent = `${data.looking_away_pct}%`;
        ids.eyes.textContent = `${data.eyes_closed_pct}%`;
        ids.phone.textContent = `${data.phone_detected_pct}%`;
        ids.people.textContent = `${data.unauthorized_people_pct}%`;

        ids.status.textContent = data.running ? "Running" : "Stopped";
        ids.status.classList.toggle("stopped", !data.running);
    } catch (_) {
        // Skip transient polling errors.
    }
}

ids.stopBtn?.addEventListener("click", async () => {
    try {
        await fetch("/realtime/stop", { method: "POST" });
        if (ids.feed) {
            ids.feed.src = "";
        }
    } finally {
        refreshScores();
    }
});

ids.restartBtn?.addEventListener("click", async () => {
    try {
        await fetch("/realtime/restart", { method: "POST" });
        if (ids.feed) {
            ids.feed.src = `/realtime/feed?ts=${Date.now()}`;
        }
    } finally {
        refreshScores();
    }
});

refreshScores();
setInterval(refreshScores, 1200);
