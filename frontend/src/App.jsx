
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Bot,
  Copy,
  Check,
  Send,
  Sparkles,
} from "lucide-react";

import "./index.css";

const API_URL = "http://192.168.5.90:8000";

const suggestions = [
  "چطور یک سند حسابداری جدید ایجاد کنم؟",
  "گزارش تحلیلی تنخواه کجاست؟",
  "چطور تنخواه تعریف کنم؟",
  "منوی سند حسابداری کجاست؟",
];

/*
 * Wrap technical / Latin content so browsers don't mix
 * Persian RTL text with English LTR text incorrectly.
 */
function BidiText({ children }) {
  return (
    <bdi dir="auto">
      {children}
    </bdi>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const textareaRef = useRef(null);
  const bottomRef = useRef(null);

  // ----------------------------------------------------------
  // Auto scroll
  // ----------------------------------------------------------

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages]);

  // ----------------------------------------------------------
  // Textarea auto resize
  // ----------------------------------------------------------

  const handleInputChange = (event) => {
    const value = event.target.value;

    setInput(value);

    const textarea = event.target;

    textarea.style.height = "auto";

    textarea.style.height =
      `${Math.min(textarea.scrollHeight, 180)}px`;
  };

  // ----------------------------------------------------------
  // Send message
  // ----------------------------------------------------------

  const sendMessage = async (text = input) => {
    const message = text.trim();

    if (!message || loading) {
      return;
    }

    setInput("");

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: message,
      },
      {
        role: "assistant",
        content: "",
      },
    ]);

    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
          }),
        }
      );

      console.log(
        "Response status:",
        response.status
      );

      console.log(
        "Response content type:",
        response.headers.get("content-type")
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      if (!response.body) {
        throw new Error(
          "Response body is not available."
        );
      }

      const reader =
        response.body.getReader();

      const decoder =
        new TextDecoder("utf-8");

      let assistantText = "";

      while (true) {
        const {
          value,
          done,
        } = await reader.read();

        if (done) {
          break;
        }

        const chunk =
          decoder.decode(
            value,
            {
              stream: true,
            }
          );

        console.log(
          "Received chunk:",
          chunk
        );

        assistantText += chunk;

        setMessages((previous) => {
          if (previous.length === 0) {
            return previous;
          }

          const updated = [
            ...previous,
          ];

          const lastIndex =
            updated.length - 1;

          updated[lastIndex] = {
            ...updated[lastIndex],
            role: "assistant",
            content: assistantText,
          };

          return updated;
        });
      }

      const remaining =
        decoder.decode();

      if (remaining) {
        assistantText += remaining;

        setMessages((previous) => {
          if (previous.length === 0) {
            return previous;
          }

          const updated = [
            ...previous,
          ];

          const lastIndex =
            updated.length - 1;

          updated[lastIndex] = {
            ...updated[lastIndex],
            role: "assistant",
            content: assistantText,
          };

          return updated;
        });
      }

      console.log(
        "Final assistant response:",
        assistantText
      );

    } catch (error) {
      console.error(
        "Chat request failed:",
        error
      );

      setMessages((previous) => {
        if (previous.length === 0) {
          return previous;
        }

        const updated = [
          ...previous,
        ];

        const lastIndex =
          updated.length - 1;

        updated[lastIndex] = {
          role: "assistant",
          error: true,
          content:
            "ارتباط با WebERP AI برقرار نشد. لطفاً مطمئن شوید که سرور FastAPI در حال اجرا است.",
        };

        return updated;
      });

    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------------------------
  // Enter key
  // ----------------------------------------------------------

  const handleKeyDown = (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      sendMessage();
    }
  };

  // ----------------------------------------------------------
  // Copy answer
  // ----------------------------------------------------------

  const copyMessage = async (
    content,
    index
  ) => {
    try {
      await navigator.clipboard.writeText(
        content
      );

      setCopiedIndex(index);

      setTimeout(() => {
        setCopiedIndex(null);
      }, 1500);

    } catch (error) {
      console.error(
        "Copy failed:",
        error
      );
    }
  };

  // ----------------------------------------------------------
  // Markdown rendering
  // ----------------------------------------------------------

  const renderMarkdown = (content) => (
    <ReactMarkdown
      components={{
        p: ({ children }) => (
          <p dir="rtl">
            <BidiText>
              {children}
            </BidiText>
          </p>
        ),

        li: ({ children }) => (
          <li dir="rtl">
            <BidiText>
              {children}
            </BidiText>
          </li>
        ),

        strong: ({ children }) => (
          <strong dir="auto">
            {children}
          </strong>
        ),

        em: ({ children }) => (
          <em dir="auto">
            {children}
          </em>
        ),

        code: ({
          inline,
          children,
          ...props
        }) => {
          if (inline) {
            return (
              <code
                dir="ltr"
                {...props}
              >
                {children}
              </code>
            );
          }

          return (
            <code
              dir="ltr"
              {...props}
            >
              {children}
            </code>
          );
        },

        pre: ({ children }) => (
          <pre
            dir="ltr"
            style={{
              textAlign: "left",
              direction: "ltr",
            }}
          >
            {children}
          </pre>
        ),

        a: ({
          children,
          href,
          ...props
        }) => (
          <a
            href={href}
            dir="ltr"
            {...props}
          >
            {children}
          </a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );

  // ----------------------------------------------------------
  // Render
  // ----------------------------------------------------------

  return (
    <div
      className="app"
      dir="rtl"
    >

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header
        className="header"
        dir="rtl"
      >

        <div className="brand">

          <div className="brand-icon">
            <Sparkles
              size={19}
              strokeWidth={2}
            />
          </div>

          <div className="brand-text">

            <div className="brand-name">
              <BidiText>
                WebERP AI
              </BidiText>
            </div>

            <div className="brand-status">

              <span className="status-dot" />

              <BidiText>
                آنلاین
              </BidiText>

            </div>

          </div>

        </div>

      </header>


      {/* ======================================================
          MAIN
      ====================================================== */}

      <main
        className="main"
        dir="rtl"
      >

        {/* ====================================================
            WELCOME
        ==================================================== */}

        {messages.length === 0 && (

          <section
            className="welcome"
            dir="rtl"
          >

            <div className="welcome-icon">

              <Bot
                size={32}
                strokeWidth={1.8}
              />

            </div>

            <h1>
              چطور می‌توانم کمکتان کنم؟
            </h1>

            <p>
              درباره منوها، امکانات و
              عملیات{" "}
              <bdi dir="ltr">
                WebERP
              </bdi>{" "}
              از من بپرسید.
            </p>

            <div className="suggestions">

              {suggestions.map(
                (suggestion) => (

                  <button
                    key={suggestion}
                    className="suggestion"
                    onClick={() =>
                      sendMessage(
                        suggestion
                      )
                    }
                    dir="rtl"
                  >

                    <span>
                      {suggestion}
                    </span>

                  </button>

                )
              )}

            </div>

          </section>

        )}


        {/* ====================================================
            CHAT
        ==================================================== */}

        {messages.length > 0 && (

          <section
            className="chat"
            dir="rtl"
          >

            {messages.map(
              (message, index) => (

                <div
                  key={index}
                  className={
                    `message-row ${
                      message.role === "user"
                        ? "user"
                        : "assistant"
                    }`
                  }
                >

                  {/* ==========================================
                      USER MESSAGE
                  ========================================== */}

                  {message.role === "user" && (

                    <div
                      className="user-message"
                      dir="auto"
                    >

                      <BidiText>
                        {message.content}
                      </BidiText>

                    </div>

                  )}


                  {/* ==========================================
                      ASSISTANT MESSAGE
                  ========================================== */}

                  {message.role === "assistant" && (

                    <>
                      <div className="avatar">

                        <Bot
                          size={17}
                          strokeWidth={2}
                        />

                      </div>

                      <div className="message-content">

                        <div
                          className={
                            `assistant-message ${
                              message.error
                                ? "error-message"
                                : ""
                            }`
                          }
                          dir="rtl"
                        >

                          {message.content ? (

                            renderMarkdown(
                              message.content
                            )

                          ) : (

                            loading &&
                            index ===
                              messages.length - 1 && (

                              <div className="thinking">
                                <span />
                                <span />
                                <span />
                              </div>

                            )

                          )}

                          {/* Streaming cursor */}

                          {loading &&
                            index ===
                              messages.length - 1 &&
                            message.content && (

                              <span
                                style={{
                                  display:
                                    "inline-block",
                                  marginRight:
                                    "3px",
                                  animation:
                                    "blink 1s infinite",
                                }}
                              >
                                ▌
                              </span>

                            )}

                          {/* Copy */}

                          {message.content &&
                            !loading && (

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

                                {copiedIndex ===
                                index ? (

                                  <Check size={15} />

                                ) : (

                                  <Copy size={15} />

                                )}

                              </button>

                            )}

                        </div>

                      </div>

                    </>

                  )}

                </div>

              )
            )}

            <div ref={bottomRef} />

          </section>

        )}

      </main>


      {/* ======================================================
          INPUT
      ====================================================== */}

      <footer
        className="input-area"
        dir="rtl"
      >

        <div
          className="input-wrapper"
          dir="rtl"
        >

          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="پیام خود را درباره WebERP بنویسید..."
            rows={1}
            disabled={loading}
            dir="rtl"
          />

          <button
            className="send-button"
            onClick={() =>
              sendMessage()
            }
            disabled={
              !input.trim() ||
              loading
            }
            title="ارسال"
          >

            <Send
              size={19}
              strokeWidth={2}
            />

          </button>

        </div>

        <div
          className="input-hint"
          dir="rtl"
        >
          WebERP AI ممکن است در بعضی موارد
          به اطلاعات بیشتری نیاز داشته باشد.
        </div>

      </footer>

    </div>
  );
}

export default App;
