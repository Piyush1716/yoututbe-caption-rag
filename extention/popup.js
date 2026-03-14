// popup.js — All logic for the extension popup

const API_BASE = "http://localhost:8000";

// ── DOM Elements ──────────────────────────────────────────────────────────────
const statusBadge    = document.getElementById("status-badge");
const statusText     = document.getElementById("status-text");
const videoIdEl      = document.getElementById("video-id");
const indexBtn       = document.getElementById("index-btn");
const chatSection    = document.getElementById("chat-section");
const messagesEl     = document.getElementById("messages");
const questionInput  = document.getElementById("question-input");
const sendBtn        = document.getElementById("send-btn");
const notYoutube     = document.getElementById("not-youtube");
const mainContent    = document.getElementById("main-content");

let currentVideoId = null;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await detectVideo();
});

async function detectVideo() {
  setStatus("detecting", "Detecting video...");

  // Ask content script for the current video ID
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab.url || !tab.url.includes("youtube.com/watch")) {
    mainContent.style.display = "none";
    notYoutube.style.display  = "flex";
    return;
  }

  let videoId = null;
  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_VIDEO_ID" });
    videoId = response?.videoId;
  } catch (e) {
    // Content script might not be ready yet — parse URL directly as fallback
    const url = new URL(tab.url);
    videoId = url.searchParams.get("v");
  }

  if (!videoId) {
    setStatus("error", "Could not detect video ID");
    return;
  }

  currentVideoId = videoId;
  videoIdEl.textContent = videoId;
  await checkIndexStatus(videoId);
}

async function checkIndexStatus(videoId) {
  setStatus("checking", "Checking index status...");
  try {
    const res  = await fetch(`${API_BASE}/status/${videoId}`);
    const data = await res.json();

    if (data.indexed) {
      setStatus("ready", "Ready to chat");
      showChat();
    } else {
      setStatus("not-indexed", "Video not indexed yet");
      indexBtn.style.display = "flex";
    }
  } catch (e) {
    setStatus("error", "Cannot reach server. Is it running?");
    console.error("[YT RAG]", e);
  }
}

// ── Index Button ──────────────────────────────────────────────────────────────
indexBtn.addEventListener("click", async () => {
  if (!currentVideoId) return;

  indexBtn.disabled    = true;
  indexBtn.textContent = "Indexing...";
  setStatus("indexing", "Fetching & indexing video...");

  try {
    const res  = await fetch(`${API_BASE}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId })
    });
    const data = await res.json();

    if (res.ok) {
      setStatus("ready", "Indexed! Ready to chat");
      indexBtn.style.display = "none";
      showChat();
      addMessage("assistant", `✅ Video indexed! Ask me anything about this video.`);
    } else {
      setStatus("error", data.detail || "Indexing failed");
      indexBtn.disabled    = false;
      indexBtn.textContent = "⚡ Index This Video";
    }
  } catch (e) {
    setStatus("error", "Server error during indexing");
    indexBtn.disabled    = false;
    indexBtn.textContent = "⚡ Index This Video";
    console.error("[YT RAG]", e);
  }
});

// ── Chat ──────────────────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || !currentVideoId) return;

  addMessage("user", question);
  questionInput.value = "";
  sendBtn.disabled    = true;

  const thinkingId = addMessage("assistant", "Thinking...", true);

  try {
    const res  = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId, question })
    });
    const data = await res.json();

    removeMessage(thinkingId);
    if (res.ok) {
      addMessage("assistant", data.answer);
    } else {
      addMessage("assistant", `❌ Error: ${data.detail || "Something went wrong"}`);
    }
  } catch (e) {
    removeMessage(thinkingId);
    addMessage("assistant", "❌ Could not reach server. Make sure it's running.");
    console.error("[YT RAG]", e);
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showChat() {
  chatSection.style.display = "flex";
}

function setStatus(type, text) {
  statusText.textContent    = text;
  statusBadge.className     = `status-dot ${type}`;
}

let msgCounter = 0;
function addMessage(role, text, isTemp = false) {
  const id  = `msg-${++msgCounter}`;
  const div = document.createElement("div");
  div.className  = `message ${role}`;
  div.id         = id;
  if (isTemp) div.classList.add("thinking");

  const bubble = document.createElement("div");
  bubble.className   = "bubble";
  bubble.textContent = text;
  div.appendChild(bubble);

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return id;
}

function removeMessage(id) {
  document.getElementById(id)?.remove();
}
