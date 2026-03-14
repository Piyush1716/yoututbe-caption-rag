// content.js — Injected into YouTube pages
// Detects the current video ID and listens for messages from popup

function getVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("v") || null;
}

// Listen for popup asking "what video is playing?"
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_VIDEO_ID") {
    const videoId = getVideoId();
    console.log("[YT RAG] Video ID detected:", videoId);
    sendResponse({ videoId });
  }
});
