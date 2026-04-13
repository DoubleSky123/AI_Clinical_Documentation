import { useState, useRef, useEffect } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';

const SAMPLE = `Doctor: What brings you in today?
Patient: I've had a fever of 101.5, sore throat, and body aches for 2 days. Also really tired.
Doctor: Any COVID exposure recently?
Patient: Yes, my coworker tested positive 4 days ago. I also have Type 2 diabetes on metformin.
Doctor: Rapid COVID test: positive. Flu: negative. Given your diabetes you qualify for Paxlovid. O2 sat is 98%. Isolate for 5 days, rest, fluids, acetaminophen for fever.`;

const WELCOME = {
  id: 'welcome',
  role: 'ai',
  type: 'text',
  content: "Hello, Doctor. Paste a patient encounter transcript or upload an audio recording, and I'll generate a structured SOAP note for you.",
};

export default function App() {
  const [messages, setMessages]           = useState([WELCOME]);
  const [inputText, setInputText]         = useState('');
  const [isProcessing, setIsProcessing]   = useState(false);
  const [isRecording, setIsRecording]     = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [error, setError]                 = useState('');

  const bottomRef    = useRef(null);
  const mediaRef     = useRef(null);
  const chunksRef    = useRef([]);
  const timerRef     = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef  = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  function addMessage(msg) {
    setMessages(prev => [...prev, { id: Date.now() + Math.random(), ...msg }]);
  }

  // ── Text transcript ───────────────────────────────────────────────────────

  async function submitTranscript(transcript) {
    addMessage({ role: 'doctor', type: 'transcript', content: transcript });
    setIsProcessing(true);
    setError('');
    try {
      const res  = await fetch('/api/v1/notes/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ transcript, patient_id: 'demo' }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');
      addMessage({
        role:    'ai',
        type:    'soap',
        content: data.soap_note,
        stats:   { total_latency_ms: data.latency_ms },
      });
    } catch (e) {
      setError(e.message);
      addMessage({ role: 'ai', type: 'text', content: `⚠ ${e.message}` });
    } finally {
      setIsProcessing(false);
    }
  }

  function handleSend() {
    const text = inputText.trim();
    if (!text || isProcessing) return;
    submitTranscript(text);
    setInputText('');
    textareaRef.current?.focus();
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSend();
  }

  // ── Audio ─────────────────────────────────────────────────────────────────

  async function submitAudio(blob, filename = 'encounter.webm') {
    addMessage({ role: 'doctor', type: 'audio', content: filename });
    setIsProcessing(true);
    setError('');
    try {
      const fd = new FormData();
      fd.append('audio', blob, filename);
      const res  = await fetch('/api/v1/voice/transcribe-and-document', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');
      addMessage({
        role:       'ai',
        type:       'soap',
        content:    data.soap_note,
        transcript: data.transcript,
        stats: {
          asr_latency_ms:   data.asr_latency_ms,
          llm_latency_ms:   data.llm_latency_ms,
          total_latency_ms: data.total_latency_ms,
        },
      });
    } catch (e) {
      setError(e.message);
      addMessage({ role: 'ai', type: 'text', content: `⚠ ${e.message}` });
    } finally {
      setIsProcessing(false);
    }
  }

  // ── Microphone ────────────────────────────────────────────────────────────

  async function toggleRecording() {
    if (isRecording) {
      clearInterval(timerRef.current);
      mediaRef.current?.stop();
      mediaRef.current?.stream?.getTracks().forEach(t => t.stop());
      setIsRecording(false);
      setRecordSeconds(0);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mr     = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        chunksRef.current = [];
        mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
        mr.onstop = () => submitAudio(
          new Blob(chunksRef.current, { type: 'audio/webm' }), 'encounter.webm'
        );
        mr.start(250);
        mediaRef.current = mr;
        setRecordSeconds(0);
        setIsRecording(true);
        timerRef.current = setInterval(() => setRecordSeconds(s => s + 1), 1000);
      } catch {
        setError('Microphone access denied. Allow microphone in browser settings.');
      }
    }
  }

  function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    submitAudio(file, file.name);
    e.target.value = '';
  }

  const fmt = s =>
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  return (
    <div className="chat-app">

      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <div className="avatar ai-avatar">✚</div>
          <div>
            <div className="header-name">AI Clinical Assistant</div>
            <div className="header-sub">LangGraph · FAISS · Qwen 2.5 · Whisper</div>
          </div>
        </div>
        <button
          className="sample-btn"
          onClick={() => setInputText(SAMPLE)}
          disabled={isProcessing}
        >
          Load sample
        </button>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map(msg => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {isProcessing && (
          <div className="chat-msg ai">
            <div className="avatar ai-avatar">✚</div>
            <div className="bubble ai typing-bubble">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        {isRecording && (
          <div className="recording-bar">
            <span className="rec-dot" />
            Recording {fmt(recordSeconds)} — click ⏹ to stop and generate note
          </div>
        )}
        {error && <div className="error-bar">⚠ {error}</div>}
        <div className="input-row">
          <button
            className={`icon-btn ${isRecording ? 'rec-active' : ''}`}
            onClick={toggleRecording}
            disabled={isProcessing}
            title={isRecording ? 'Stop recording' : 'Record encounter'}
          >
            {isRecording ? '⏹' : '🎙'}
          </button>
          <button
            className="icon-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing || isRecording}
            title="Upload audio file"
          >
            📎
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,.webm,.mp4,.ogg"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="Paste physician-patient transcript…  (Ctrl+Enter to send)"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isProcessing || isRecording}
            rows={3}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={isProcessing || !inputText.trim() || isRecording}
            title="Send (Ctrl+Enter)"
          >
            ➤
          </button>
        </div>
      </div>

    </div>
  );
}
