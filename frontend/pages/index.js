import { useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      if (res.ok) {
        setResponse(data.response);
      } else {
        setResponse(`Error: ${data.detail || res.statusText}`);
      }
    } catch (err) {
      setResponse("Error: " + err.message);
    }
    setLoading(false);
  };

  const isError = response.startsWith("Error:");

  return (
    <div className="home-container">
      <div className="home-overlay" />
      <div className="home-content">
        <h1 className="home-title">Sentiment Analysis Agent</h1>

        <div className="glass">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type a sentence to analyse sentiment..."
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
          />
          <button onClick={sendMessage} disabled={loading}>
            {loading ? "Analysing..." : "Send"}
          </button>
        </div>

        {response && (
          <div className="glass response-panel">
            <div className="response-label">Agent</div>
            <div className={`response-text${isError ? " error" : ""}`}>
              {response}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}