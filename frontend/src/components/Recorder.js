import { useState, useRef, useEffect } from 'react';

const EMPTY_SOAP = {
  subjective: '', objective: '', assessment: '', plan: '',
  icd10_codes: [], medications: [], follow_up: '', clinical_flags: []
};

export default function Recorder({ onResult, onError, onProcessing }) {
  const [status, setStatus]   = useState('idle');      // idle | recording | processing | done | error
  const [seconds, setSeconds] = useState(0);

  const mediaRef    = useRef(null);
  const chunksRef   = useRef([]);
  const timerRef    = useRef(null);

  // Cleanup on unmount
  useEffect(() => () => {
    clearInterval(timerRef.current);
    mediaRef.current?.stream?.getTracks().forEach(t => t.stop());
  }, []);

  const fmt = s => `${String(Math.floor(s / 60)).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr     = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunksRef.current = [];

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => handleUpload(new Blob(chunksRef.current, { type: 'audio/webm' }));

      mr.start(250);
      mediaRef.current = mr;
      setSeconds(0);
      setStatus('recording');
      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000);
    } catch {
      onError('Microphone access denied. Please allow microphone in browser settings.');
      setStatus('error');
    }
  }

  function stopRecording() {
    clearInterval(timerRef.current);
    mediaRef.current?.stop();
    mediaRef.current?.stream?.getTracks().forEach(t => t.stop());
    setStatus('processing');
    onProcessing(true);
  }

  async function handleUpload(blob) {
    const fd = new FormData();
    fd.append('audio', blob, 'encounter.webm');
    try {
      const res  = await fetch('/api/v1/voice/transcribe-and-document', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server error');
      onResult(data);
      setStatus('done');
    } catch (e) {
      onError(e.message);
      setStatus('error');
    } finally {
      onProcessing(false);
    }
  }

  const isRecording  = status === 'recording';
  const isProcessing = status === 'processing';

  const btnClass = `record-btn ${isRecording ? 'recording' : isProcessing ? 'processing' : 'idle'}`;
  const icon     = isRecording ? '⏹' : isProcessing ? '⏳' : '🎙';

  const statusMsg = {
    idle:       'Press to start recording',
    recording:  'Recording… press to stop',
    processing: 'Transcribing & generating note…',
    done:       'SOAP note generated ✓',
    error:      'Error — try again',
  }[status];

  return (
    <div className="recorder-center">
      <button
        className={btnClass}
        onClick={isRecording ? stopRecording : isProcessing ? null : startRecording}
        title={statusMsg}
      >
        {icon}
      </button>

      <div className="timer">{isRecording ? fmt(seconds) : ''}</div>

      <p className={`status-text ${status}`}>{statusMsg}</p>
    </div>
  );
}
