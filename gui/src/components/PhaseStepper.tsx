import React, { useState } from 'react';
import { ConfirmDialog } from './ConfirmDialog';

const PHASES = [
  { key: 'IDEA_GENERATION', label: '1. Ideas', desc: 'Brainstorm & seed topic selection', color: 'var(--phase-idea)' },
  { key: 'WAIT_FOR_GEMINI_REPORT', label: '2. Research', desc: 'Deep research dossier collection', color: 'var(--phase-research)' },
  { key: 'SYNTHESIS', label: '3. Synthesis', desc: 'Narrative arc & structure breakdown', color: 'var(--phase-synthesis)' },
  { key: 'EVIDENCE_GATHERING', label: '4. Evidence', desc: 'Image & video asset downloading', color: 'var(--phase-evidence)' },
  { key: 'SCRIPTING', label: '5. Scripting', desc: 'Script drafting & voiceover timing', color: 'var(--phase-scripting)' },
  { key: 'VIDEO_RENDER', label: '6. Render', desc: 'TTS synthesis & final video editing', color: 'var(--phase-render)' },
];

interface PhaseStepperProps {
  currentPhase: string;
  onSelectPhase?: (phase: string) => void;
}

export const PhaseStepper: React.FC<PhaseStepperProps> = ({ currentPhase, onSelectPhase }) => {
  const currentIndex = PHASES.findIndex((p) => p.key === currentPhase);
  const [targetOverridePhase, setTargetOverridePhase] = useState<string | null>(null);

  const handleStepClick = (phaseKey: string) => {
    if (phaseKey === currentPhase || !onSelectPhase) return;
    setTargetOverridePhase(phaseKey);
  };

  const confirmOverride = () => {
    if (targetOverridePhase && onSelectPhase) {
      onSelectPhase(targetOverridePhase);
    }
    setTargetOverridePhase(null);
  };

  const progressPercent = currentIndex >= 0 ? (currentIndex / (PHASES.length - 1)) * 100 : 0;

  return (
    <>
      <div className="glass-card" style={{ padding: '24px 20px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative' }}>
          {/* Background connecting track */}
          <div
            style={{
              position: 'absolute',
              top: '19px',
              left: '5%',
              right: '5%',
              height: '3px',
              background: 'var(--bg-tertiary)',
              zIndex: 1,
            }}
          >
            {/* Filled progress bar */}
            <div
              style={{
                height: '100%',
                width: `${progressPercent}%`,
                background: 'var(--accent-gradient)',
                transition: 'width 0.4s ease',
              }}
            />
          </div>

          {PHASES.map((phase, idx) => {
            const isCompleted = idx < currentIndex;
            const isCurrent = idx === currentIndex;

            return (
              <div
                key={phase.key}
                title={`${phase.label}: ${phase.desc} (Click to jump)`}
                onClick={() => handleStepClick(phase.key)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '8px',
                  cursor: onSelectPhase ? 'pointer' : 'default',
                  zIndex: 2,
                  flex: 1,
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    width: '38px',
                    height: '38px',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 700,
                    fontSize: '0.9rem',
                    background: isCurrent
                      ? phase.color
                      : isCompleted
                      ? 'rgba(255, 255, 255, 0.15)'
                      : 'var(--bg-tertiary)',
                    color: isCurrent ? '#000' : isCompleted ? '#fff' : 'var(--text-muted)',
                    border: isCurrent
                      ? `3px solid #fff`
                      : isCompleted
                      ? `2px solid ${phase.color}`
                      : '1px solid var(--border-subtle)',
                    boxShadow: isCurrent ? `0 0 16px ${phase.color}` : 'none',
                    animation: isCurrent ? 'pulseGlow 2s infinite' : 'none',
                    transition: 'all 0.3s ease',
                  }}
                >
                  {isCompleted ? '✓' : idx + 1}
                </div>

                <div style={{ textAlign: 'center' }}>
                  <span
                    style={{
                      fontSize: '0.8rem',
                      display: 'block',
                      fontWeight: isCurrent ? 700 : 500,
                      color: isCurrent ? phase.color : isCompleted ? 'var(--text-primary)' : 'var(--text-muted)',
                    }}
                  >
                    {phase.label}
                  </span>
                  <span
                    style={{
                      fontSize: '0.7rem',
                      color: 'var(--text-muted)',
                      display: 'block',
                      marginTop: '2px',
                    }}
                  >
                    {phase.desc}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <ConfirmDialog
        isOpen={!!targetOverridePhase}
        title="Jump Pipeline Phase?"
        message={`Override current phase to '${PHASES.find((p) => p.key === targetOverridePhase)?.label}'? This will force the state machine to run from this stage.`}
        confirmLabel="Jump Phase"
        variant="warning"
        onConfirm={confirmOverride}
        onCancel={() => setTargetOverridePhase(null)}
      />
    </>
  );
};
