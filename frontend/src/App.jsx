import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ArrowUp,
  Bot,
  Copy,
  Check,
  Sparkles,
} from "lucide-react";

import "./index.css";

const API_URL = "http://127.0.0.1:8000";

const suggestions = [
  "چطور یک سند حسابداری جدید ایجاد کنم؟",
  "گزارش تحلیلی تنخواه کجاست؟",
  "چطور تنخواه تعریف کنم؟",
  "منوی سند حسابداری کجاست؟",
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const textareaRef = useRef(null);
  const bottomRef = useRef(null);

  /*
   * Scroll to the newest message.
   */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  /*
   * Automatically resize textarea.
   */
  const resizeTextarea = () => {
    const textarea = textareaRef.current;

    if (!textarea) {
      return;
    }

    textarea.style.height = "auto";
    textarea.style.height =
      Math.min(textarea.scrollHeight, 180) + "px";
  };

  /*
   * Send message to FastAPI.
   */
  const sendMessage = async (text = input) => {
    const message = text.trim();

    if (!message || loading) {
      return;
    }

    setInput("");

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const userMessage = {
      role: "user",
      content: message,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
    ]);

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      const assistantMessage = {
        role: "assistant",
        content:
          data.answer ||
          "متأسفانه پاسخی دریافت نشد.",
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);
    } catch (error) {
      console.error(error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          error: true,
          content:
            "ارتباط با WebERP AI برقرار نشد. لطفاً مطمئن شوید که سرور FastAPI در حال اجرا است.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /*
   * Enter = send
   * Shift + Enter = new line
   */
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  /*
   * Copy assistant answer.
   */
  const copyMessage = async (content, index) => {
    try {
      await navigator.clipboard.writeText(content);

      setCopiedIndex(index);

      setTimeout(() => {
        setCopiedIndex(null);
      }, 1500);
    } catch (error) {
      console.error(error);
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={19} strokeWidth={2.2} />
          </div>

          <div className="brand-text">
            <span className="brand-name">
              WebERP AI
            </span>

            <span className="brand-status">
              <span className="status-dot" />
              آماده پاسخگویی
            </span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="main">
        {isEmpty ? (
          <section className="welcome">
            <div className="welcome-icon">
              <Bot size={30} strokeWidth={1.8} />
            </div>

            <h1>
              سلام، چطور می‌تونم کمکت کنم؟
            </h1>

            <p>
              درباره منوها و امکانات WebERP
              سؤال بپرسید.
            </p>

            <div className="suggestions">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  className="suggestion"
                  onClick={() =>
                    sendMessage(suggestion)
                  }
                >
                  <span>{suggestion}</span>
                  <ArrowUp
                    size={16}
                    className="suggestion-arrow"
                  />
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="chat">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message-row ${message.role}`}
              >
                {message.role === "assistant" && (
                  <div className="avatar">
                    <Sparkles size={17} />
                  </div>
                )}

                <div className="message-content">
                  {message.role === "assistant" ? (
                    <div
                      className={`assistant-message ${
                        message.error
                          ? "error-message"
                          : ""
                      }`}
                      dir="rtl"
                    >
                      <ReactMarkdown>
                        {message.content}
                      </ReactMarkdown>

                      {!message.error && (
                        <button
                          className="copy-button"
                          onClick={() =>
                            copyMessage(
                              message.content,
                              index
                            )
                          }
                          title="کپی پاسخ"
                        >
                          {copiedIndex === index ? (
                            <Check size={15} />
                          ) : (
                            <Copy size={15} />
                          )}
                        </button>
                      )}
                    </div>
                  ) : (
                    <div
                      className="user-message"
                      dir="rtl"
                    >
                      {message.content}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row assistant">
                <div className="avatar">
                  <Sparkles size={17} />
                </div>

                <div className="thinking">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </section>
        )}
      </main>

      {/* Input */}
      <div className="input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            value={input}
            placeholder="پیام خود را درباره WebERP بنویسید..."
            rows={1}
            disabled={loading}
            onChange={(event) => {
              setInput(event.target.value);
              resizeTextarea();
            }}
            onKeyDown={handleKeyDown}
          />

          <button
            className="send-button"
            disabled={
              !input.trim() || loading
            }
            onClick={() => sendMessage()}
            title="ارسال"
          >
            <ArrowUp size={19} strokeWidth={2.4} />
          </button>
        </div>

        <div className="input-hint">
          WebERP AI ممکن است در برخی پاسخ‌ها
          نیاز به بررسی اطلاعات داشته باشد.
        </div>
      </div>
    </div>
  );
}

export default App;