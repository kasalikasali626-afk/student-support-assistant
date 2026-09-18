import sys
import os
import warnings
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. RAG Core Setup ---
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "regulations.txt")
loader = TextLoader(DATA_FILE)
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="llama3.2", temperature=0.2)

contextualize_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in the regulations.")
])
history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a College Support Assistant. Answer using the provided context.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
doc_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_retriever, doc_chain)

# --- 2. Human-Made Web UI Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Student Support Portal</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f4f6f9;
            color: #212529;
            line-height: 1.5;
        }
        .header {
            background-color: #0b3954;
            color: #ffffff;
            padding: 16px 24px;
            border-bottom: 3px solid #087e8b;
        }
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.2px; }
        .header p { font-size: 13px; color: #bfdbfe; margin-top: 2px; }
        .header-badge {
            background: #ffffff1a;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            border: 1px solid #ffffff33;
        }
        .main-container {
            max-width: 1200px;
            margin: 24px auto;
            padding: 0 16px;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 24px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .card-header {
            padding: 14px 18px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            font-size: 14px;
            font-weight: 600;
            color: #343a40;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-body { padding: 18px; }
        .rule-item {
            padding: 10px 12px;
            margin-bottom: 10px;
            background: #f8fafc;
            border-left: 3px solid #087e8b;
            border-radius: 0 4px 4px 0;
            font-size: 13px;
        }
        .rule-item strong { display: block; color: #0b3954; margin-bottom: 2px; }
        .chip-group {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        .chip {
            background: #e9ecef;
            border: 1px solid #ced4da;
            padding: 6px 10px;
            font-size: 12px;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .chip:hover {
            background: #0b3954;
            color: #ffffff;
            border-color: #0b3954;
        }
        .chat-box {
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.5;
        }
        .msg-user {
            align-self: flex-end;
            background-color: #0b3954;
            color: #ffffff;
            border-bottom-right-radius: 1px;
        }
        .msg-assistant {
            align-self: flex-start;
            background-color: #f1f5f9;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 1px;
        }
        .msg-meta {
            font-size: 11px;
            margin-bottom: 4px;
            opacity: 0.8;
            font-weight: 500;
        }
        .chat-input-area {
            padding: 14px 18px;
            border-top: 1px solid #dee2e6;
            background: #f8f9fa;
        }
        .input-row {
            display: flex;
            gap: 10px;
        }
        .input-row input {
            flex: 1;
            padding: 10px 14px;
            font-size: 14px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            outline: none;
        }
        .input-row input:focus {
            border-color: #087e8b;
            box-shadow: 0 0 0 2px rgba(8,126,139,0.15);
        }
        .btn-submit {
            background-color: #087e8b;
            color: #ffffff;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
        }
        .btn-submit:hover { background-color: #066570; }
        .btn-submit:disabled { background-color: #adb5bd; cursor: not-allowed; }
        .footer {
            max-width: 1200px;
            margin: 20px auto;
            text-align: center;
            font-size: 12px;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div>
                <h1>College Student Affairs & Academic Information Desk</h1>
                <p>Official Regulations & Student Assistance Portal (Academic Year 2024-25)</p>
            </div>
            <div class="header-badge">
                Ollama Engine: llama3.2 | Status: Connected
            </div>
        </div>
    </div>

    <div class="main-container">
        <!-- Sidebar with Official Bulletins -->
        <div class="card">
            <div class="card-header">
                Official Regulations Summary
            </div>
            <div class="card-body">
                <div class="rule-item">
                    <strong>Attendance Policy</strong>
                    Minimum 75% attendance is required to appear for semester exams.
                </div>
                <div class="rule-item">
                    <strong>Exam Evaluation Pattern</strong>
                    Internal assessments: 40 marks | Final examination: 60 marks.
                </div>
                <div class="rule-item">
                    <strong>Hostel Curfew Rules</strong>
                    Hostel in-time is 8:30 PM sharp. Late entry requires Warden approval.
                </div>
                <div class="rule-item">
                    <strong>Semester Fee Deadline</strong>
                    Must be paid before November 15th without fine.
                </div>

                <div style="margin-top: 18px;">
                    <span style="font-size: 12px; font-weight: 600; color: #495057;">Quick Inquiries:</span>
                    <div class="chip-group">
                        <button class="chip" onclick="askQuick('What is the minimum attendance required for semester exams?')">Attendance Limit</button>
                        <button class="chip" onclick="askQuick('How are internal and external marks distributed for exams?')">Exam Pattern</button>
                        <button class="chip" onclick="askQuick('What is the hostel gate closing time and curfew policy?')">Hostel In-Time</button>
                        <button class="chip" onclick="askQuick('When is the last date to pay the semester fee?')">Fee Due Date</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Chat Assistant Area -->
        <div class="card chat-box">
            <div class="card-header">
                <span>Student Inquiry Assistant (RAG + Context Memory)</span>
                <button onclick="clearChat()" style="background:none; border:1px solid #ced4da; padding:3px 8px; border-radius:3px; font-size:11px; cursor:pointer; color:#6c757d;">Clear Session</button>
            </div>
            <div class="chat-history" id="chatHistory">
                <div class="message msg-assistant">
                    <div class="msg-meta">Student Support Desk</div>
                    Hello! Welcome to the Student Academic Information Desk. You can ask any question regarding college regulations, exam patterns, hostel timings, or fee payment schedules.
                </div>
            </div>
            <div class="chat-input-area">
                <form id="chatForm" onsubmit="sendMessage(event)" class="input-row">
                    <input type="text" id="userInput" placeholder="Type your academic or campus inquiry here (e.g. Can I take the exam with 72% attendance?)..." autocomplete="off" required>
                    <button type="submit" id="submitBtn" class="btn-submit">Submit Inquiry</button>
                </form>
            </div>
        </div>
    </div>

    <div class="footer">
        Office of the Dean of Academic Affairs &bull; Campus Student Welfare Cell &bull; Powered by LangChain + ChromaDB RAG
    </div>

    <script>
        const chatHistoryEl = document.getElementById('chatHistory');
        const userInputEl = document.getElementById('userInput');
        const submitBtn = document.getElementById('submitBtn');

        function appendMessage(role, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + (role === 'user' ? 'msg-user' : 'msg-assistant');
            
            const meta = document.createElement('div');
            meta.className = 'msg-meta';
            meta.innerText = role === 'user' ? 'Student' : 'Student Support Desk';
            
            const content = document.createElement('div');
            content.innerText = text;
            
            msgDiv.appendChild(meta);
            msgDiv.appendChild(content);
            chatHistoryEl.appendChild(msgDiv);
            chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
        }

        async function sendMessage(e) {
            if (e) e.preventDefault();
            const text = userInputEl.value.trim();
            if (!text) return;

            appendMessage('user', text);
            userInputEl.value = '';
            userInputEl.disabled = true;
            submitBtn.disabled = true;
            submitBtn.innerText = 'Consulting Regulations...';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: text })
                });
                const data = await res.json();
                if (data.answer) {
                    appendMessage('assistant', data.answer);
                } else {
                    appendMessage('assistant', 'Sorry, I could not process your query. Please try again.');
                }
            } catch (err) {
                appendMessage('assistant', 'Error connecting to the support server.');
            } finally {
                userInputEl.disabled = false;
                submitBtn.disabled = false;
                submitBtn.innerText = 'Submit Inquiry';
                userInputEl.focus();
            }
        }

        function askQuick(question) {
            userInputEl.value = question;
            sendMessage();
        }

        async function clearChat() {
            await fetch('/api/clear', { method: 'POST' });
            chatHistoryEl.innerHTML = `
                <div class="message msg-assistant">
                    <div class="msg-meta">Student Support Desk</div>
                    Session refreshed. How can I assist you today?
                </div>
            `;
        }
    </script>
</body>
</html>
"""

# --- 3. Flask Server & CLI Switch ---
app = Flask(__name__)
session_chat_history = []

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/query", methods=["POST"])
def api_query():
    global session_chat_history
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    res = rag_chain.invoke({"input": query, "chat_history": session_chat_history})
    answer = res["answer"]
    session_chat_history.append(HumanMessage(content=query))
    session_chat_history.append(AIMessage(content=answer))
    return jsonify({"answer": answer})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    global session_chat_history
    session_chat_history = []
    return jsonify({"status": "cleared"})

def run_cli():
    print("Student Assistant Agent Online (Type 'quit' to exit)")
    cli_history = []
    while True:
        query = input("\nStudent: ")
        if query.lower() == "quit":
            break
        res = rag_chain.invoke({"input": query, "chat_history": cli_history})
        print(f"\nAssistant: {res['answer']}")
        cli_history.append(HumanMessage(content=query))
        cli_history.append(AIMessage(content=res['answer']))

if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        port = 5001
        print(f"==========================================================")
        print(f" Campus Student Support Portal Active")
        print(f" URL: http://127.0.0.1:{port}")
        print(f" Running with LangChain + ChromaDB + Ollama (llama3.2)")
        print(f" (To run in CLI mode instead, execute: python app.py --cli)")
        print(f"==========================================================")
        Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
        app.run(host="127.0.0.1", port=port, debug=False)
