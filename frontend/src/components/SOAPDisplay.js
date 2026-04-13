export default function SOAPDisplay({ soap, transcript, stats }) {
  if (!soap) return (
    <div className="soap-text empty">
      SOAP note will appear here after recording…
    </div>
  );

  const Section = ({ label, colorClass, children }) => (
    <div className="soap-section">
      <div className={`soap-label ${colorClass}`}>{label}</div>
      {children}
    </div>
  );

  const isEmpty = v => !v || (Array.isArray(v) && v.length === 0) || v === '';

  return (
    <div>
      {/* Latency stats */}
      {stats && (
        <div className="stats-row">
          <div className="stat-chip">ASR <span>{stats.asr_latency_ms}ms</span></div>
          <div className="stat-chip">LLM <span>{stats.llm_latency_ms}ms</span></div>
          <div className="stat-chip">Total <span>{stats.total_latency_ms}ms</span></div>
        </div>
      )}

      <div style={{ marginTop: 20 }}>
        {/* S */}
        <Section label="S — Subjective" colorClass="s">
          <div className={`soap-text ${isEmpty(soap.subjective) ? 'empty' : ''}`}>
            {soap.subjective || '—'}
          </div>
        </Section>

        {/* O */}
        <Section label="O — Objective" colorClass="o">
          <div className={`soap-text ${isEmpty(soap.objective) ? 'empty' : ''}`}>
            {soap.objective || '—'}
          </div>
        </Section>

        {/* A */}
        <Section label="A — Assessment" colorClass="a">
          <div className={`soap-text ${isEmpty(soap.assessment) ? 'empty' : ''}`}>
            {soap.assessment || '—'}
          </div>
        </Section>

        {/* P */}
        <Section label="P — Plan" colorClass="p">
          <div className={`soap-text ${isEmpty(soap.plan) ? 'empty' : ''}`}>
            {soap.plan || '—'}
          </div>
        </Section>

        {/* ICD-10 Codes */}
        {!isEmpty(soap.icd10_codes) && (
          <div className="soap-section">
            <div className="soap-label s">ICD-10 Codes</div>
            <div className="tags-row">
              {soap.icd10_codes.map(c => (
                <span key={c} className="tag icd">{c}</span>
              ))}
            </div>
          </div>
        )}

        {/* Medications */}
        {!isEmpty(soap.medications) && (
          <div className="soap-section">
            <div className="soap-label o">Medications</div>
            <div className="tags-row">
              {soap.medications.map((m, i) => (
                <span key={i} className="tag med">{m}</span>
              ))}
            </div>
          </div>
        )}

        {/* Clinical Flags */}
        {!isEmpty(soap.clinical_flags) && (
          <div className="soap-section">
            <div className="soap-label" style={{ color: '#f87171' }}>⚠ Clinical Flags</div>
            <div className="tags-row">
              {soap.clinical_flags.map((f, i) => (
                <span key={i} className="tag flag">{f}</span>
              ))}
            </div>
          </div>
        )}

        {/* Follow-up */}
        {!isEmpty(soap.follow_up) && (
          <div className="soap-section">
            <div className="soap-label p">Follow-up</div>
            <div className="soap-text">{soap.follow_up}</div>
          </div>
        )}
      </div>

      {/* Raw transcript toggle */}
      {transcript && (
        <details style={{ marginTop: 16 }}>
          <summary style={{ cursor: 'pointer', fontSize: '0.78rem', color: '#64748b' }}>
            Show raw transcript
          </summary>
          <div className="transcript-box" style={{ marginTop: 8 }}>{transcript}</div>
        </details>
      )}
    </div>
  );
}
