import SOAPDisplay from './SOAPDisplay';

export default function ChatMessage({ message }) {
  const { role, type, content, transcript, stats } = message;

  if (role === 'doctor') {
    return (
      <div className="chat-msg doctor">
        <div className="bubble doctor">
          {type === 'audio'
            ? <span className="audio-label">🎙 {content}</span>
            : <pre className="transcript-pre">{content}</pre>
          }
        </div>
        <div className="avatar doctor-avatar">Dr</div>
      </div>
    );
  }

  return (
    <div className="chat-msg ai">
      <div className="avatar ai-avatar">✚</div>
      <div className="bubble ai">
        {type === 'text' && <p className="ai-text">{content}</p>}
        {type === 'soap' && (
          <SOAPDisplay soap={content} transcript={transcript} stats={stats} />
        )}
      </div>
    </div>
  );
}
