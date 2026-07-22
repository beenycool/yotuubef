import React, { useState, useEffect, useMemo } from 'react';
import * as yaml from 'js-yaml';

interface YamlBoxEditorProps {
  yamlContent: string;
  onChangeYaml: (newYamlContent: string) => void;
}

// Section metadata mapping for rich visuals & icons
const SECTION_META: Record<string, { title: string; icon: string; desc: string; color: string }> = {
  api: {
    title: 'APIs & Credentials',
    icon: '🌐',
    desc: 'NVIDIA NIM, Reddit OAuth, YouTube API keys & model definitions',
    color: 'accent-api',
  },
  video_processing: {
    title: 'Video & Rendering',
    icon: '📹',
    desc: 'Resolution, FPS, codecs, bitrates, speed optimization, and color grading',
    color: 'accent-video_processing',
  },
  audio: {
    title: 'Audio & Music',
    icon: '🎵',
    desc: 'Background music volume, audio limiter, noise gate, and sound effects',
    color: 'accent-audio',
  },
  youtube: {
    title: 'YouTube & Publishing',
    icon: '🚀',
    desc: 'Default privacy, hashtag generation, automated descriptions, & comment settings',
    color: 'accent-youtube',
  },
  tts: {
    title: 'Text-to-Speech Voice',
    icon: '🗣️',
    desc: 'Primary TTS provider & narrator speaker voice configuration',
    color: 'accent-tts',
  },
  content: {
    title: 'Subreddits & Content Filtering',
    icon: '📰',
    desc: 'Curated subreddits & demonetization / NSFW content filter lists',
    color: 'accent-content',
  },
  paths: {
    title: 'Directory & Storage Paths',
    icon: '📁',
    desc: 'Paths for assets, downloads, music, sound effects, logs, & cache',
    color: 'accent-paths',
  },
  database: {
    title: 'Database & Retention',
    icon: '💾',
    desc: 'SQLite database location & data retention cleanup intervals',
    color: 'accent-database',
  },
};

// Helper human formatting for field names
const formatLabel = (key: string): string => {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

// Predefined option definitions for configuration fields (with special emphasis on video processing)
const DROPDOWN_OPTIONS: Record<string, Array<{ label: string; value: any }> | Array<string | number>> = {
  // Video & Processing (interactive box video settings)
  video_codec: ['libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc', 'vp9', 'av1'],
  audio_codec: ['aac', 'mp3', 'opus', 'flac', 'pcm_s16le'],
  target_fps: [24, 30, 60, 120],
  video_bitrate: ['2M', '5M', '8M', '10M', '12M', '16M', '25M', '50M'],
  audio_bitrate: ['96k', '128k', '192k', '256k', '320k'],
  pixel_format: ['yuv420p', 'yuv422p', 'yuv444p', 'nv12'],
  video_quality_profile: [
    'ultrafast',
    'superfast',
    'veryfast',
    'faster',
    'fast',
    'medium',
    'slow',
    'slower',
    'veryslow',
    'speed',
    'balanced',
    'quality',
  ],
  speed_optimization_level: ['disabled', 'low', 'medium', 'high', 'aggressive'],
  max_file_size_mb: [100, 250, 500, 1000, 2000],
  max_video_duration: [60, 180, 300, 600, 1200],
  min_video_duration: [3, 5, 10, 15, 30],

  // Audio & Sound
  limiter_threshold: [-1, -2, -3, -6],
  noise_gate_threshold: [-30, -35, -40, -50],
  volume: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

  // TTS & Voice
  primary_service: ['qwen3_tts', 'gtts', 'pyttsx3', 'edge-tts', 'bark', 'coqui'],
  narrator_speaker: ['Ryan', 'Adam', 'Antoni', 'Arnold', 'Bella', 'Domi', 'Elli', 'Josh', 'Rachel', 'Sam'],

  // YouTube & Publishing
  default_privacy: ['public', 'unlisted', 'private'],
  default_category: [
    { label: '22 - People & Blogs', value: '22' },
    { label: '20 - Gaming', value: '20' },
    { label: '24 - Entertainment', value: '24' },
    { label: '28 - Science & Tech', value: '28' },
    { label: '10 - Music', value: '10' },
    { label: '27 - Education', value: '27' },
    { label: '1 - Film & Animation', value: '1' },
    { label: '23 - Comedy', value: '23' },
  ],
  thumbnail_quality: [80, 85, 90, 95, 100],

  // AI & APIs
  ai_provider: ['nvidia_nim', 'openai', 'anthropic', 'ollama', 'gemini'],
  nvidia_nim_model: ['moonshotai/kimi-k2.6', 'minimaxai/minimax-m3', 'meta/llama-3.3-70b-instruct', 'deepseek-ai/deepseek-r1', 'mistralai/mistral-large-2411'],
  nvidia_nim_alt_model: ['minimaxai/minimax-m3', 'moonshotai/kimi-k2.6', 'meta/llama-3.3-70b-instruct', 'deepseek-ai/deepseek-r1', 'mistralai/mistral-large-2411'],
};

// Resolution presets for default_output_resolution
const RESOLUTION_PRESETS = [
  { label: '📱 Vertical Shorts / TikTok (1080 x 1920)', value: [1080, 1920] },
  { label: '🖥️ Horizontal 16:9 HD (1920 x 1080)', value: [1920, 1080] },
  { label: '⏹️ Square 1:1 (1080 x 1080)', value: [1080, 1080] },
  { label: '📱 Compact Vertical (720 x 1280)', value: [720, 1280] },
  { label: '🌟 4K Ultra HD Shorts (2160 x 3840)', value: [2160, 3840] },
];

// Helper control component for rendering supported dropdown selection
const DropdownControl: React.FC<{
  value: any;
  options: Array<{ label: string; value: any }> | Array<string | number>;
  onChange: (newVal: any) => void;
}> = ({ value, options, onChange }) => {
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [customVal, setCustomVal] = useState(String(value ?? ''));

  const normalizedOptions = useMemo(() => {
    return options.map((opt) => {
      if (typeof opt === 'object' && opt !== null && 'label' in opt) {
        return opt;
      }
      return { label: String(opt), value: opt };
    });
  }, [options]);

  const valueExistsInOptions = normalizedOptions.some((opt) => String(opt.value) === String(value));

  if (isCustomMode) {
    return (
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', width: '100%' }}>
        <input
          type="text"
          value={customVal}
          onChange={(e) => {
            setCustomVal(e.target.value);
            let parsed: any = e.target.value;
            if (typeof value === 'number' && !isNaN(Number(parsed)) && parsed !== '') {
              parsed = Number(parsed);
            }
            onChange(parsed);
          }}
          placeholder="Enter custom value..."
          autoFocus
        />
        <button
          className="btn btn-secondary"
          style={{ padding: '6px 10px', fontSize: '0.75rem', whiteSpace: 'nowrap' }}
          onClick={() => setIsCustomMode(false)}
          title="Back to Dropdown Options"
        >
          📋 Presets
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', width: '100%' }}>
      <select
        value={String(value ?? '')}
        onChange={(e) => {
          const selectedStr = e.target.value;
          if (selectedStr === '__CUSTOM__') {
            setIsCustomMode(true);
            return;
          }
          const matchedOpt = normalizedOptions.find((opt) => String(opt.value) === selectedStr);
          const newVal = matchedOpt ? matchedOpt.value : selectedStr;
          onChange(newVal);
        }}
      >
        {!valueExistsInOptions && value !== undefined && value !== null && value !== '' && (
          <option value={String(value)}>Current ({String(value)})</option>
        )}
        {normalizedOptions.map((opt, idx) => (
          <option key={idx} value={String(opt.value)}>
            {opt.label}
          </option>
        ))}
        <option value="__CUSTOM__">✏️ Custom value...</option>
      </select>
    </div>
  );
};

export const YamlBoxEditor: React.FC<YamlBoxEditorProps> = ({ yamlContent, onChangeYaml }) => {
  const [viewMode, setViewMode] = useState<'boxes' | 'raw'>('boxes');
  const [parsedData, setParsedData] = useState<Record<string, any>>({});
  const [rawText, setRawText] = useState(yamlContent);
  const [parseError, setParseError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});

  // Dynamic add key states
  const [newKeySection, setNewKeySection] = useState<string | null>(null);
  const [newFieldName, setNewFieldName] = useState('');
  const [newFieldVal, setNewFieldVal] = useState('');
  const [newSectionName, setNewSectionName] = useState('');
  const [showAddSectionModal, setShowAddSectionModal] = useState(false);

  // Parse YAML on mount or when external yamlContent changes
  useEffect(() => {
    try {
      const parsed = yaml.load(yamlContent);
      if (parsed && typeof parsed === 'object') {
        setParsedData(parsed as Record<string, any>);
        setParseError(null);
      }
    } catch (err: any) {
      setParseError(err.message || 'Invalid YAML format');
    }
  }, [yamlContent]);

  // Sync internal object updates back to YAML string
  const updateYamlData = (newData: Record<string, any>) => {
    setParsedData(newData);
    try {
      const dump = yaml.dump(newData, { indent: 2, lineWidth: -1, noRefs: true });
      setRawText(dump);
      setParseError(null);
      onChangeYaml(dump);
    } catch (err: any) {
      console.error('YAML dump error:', err);
    }
  };

  // Handle updates in raw text editor
  const handleRawTextChange = (text: string) => {
    setRawText(text);
    onChangeYaml(text);
    try {
      const parsed = yaml.load(text);
      if (parsed && typeof parsed === 'object') {
        setParsedData(parsed as Record<string, any>);
        setParseError(null);
      } else {
        setParseError('YAML parsed but is not an object structure.');
      }
    } catch (err: any) {
      setParseError(err.message || 'YAML Syntax Error');
    }
  };

  // Mutate nested field value helper (js-performance: use structuredClone instead of JSON stringify overhead)
  const handleFieldChange = (path: string[], value: any) => {
    const copy = typeof structuredClone === 'function' ? structuredClone(parsedData) : JSON.parse(JSON.stringify(parsedData));
    let curr = copy;
    for (let i = 0; i < path.length - 1; i++) {
      if (!curr[path[i]]) curr[path[i]] = {};
      curr = curr[path[i]];
    }
    curr[path[path.length - 1]] = value;
    updateYamlData(copy);
  };

  // Delete field helper
  const handleDeleteField = (path: string[]) => {
    const copy = typeof structuredClone === 'function' ? structuredClone(parsedData) : JSON.parse(JSON.stringify(parsedData));
    let curr = copy;
    for (let i = 0; i < path.length - 1; i++) {
      if (!curr[path[i]]) return;
      curr = curr[path[i]];
    }
    delete curr[path[path.length - 1]];
    updateYamlData(copy);
  };

  // Add field to section helper
  const handleAddFieldToSection = (sectionKey: string) => {
    if (!newFieldName.trim()) return;
    const path = [sectionKey, newFieldName.trim()];
    let val: any = newFieldVal.trim();
    if (val === 'true') val = true;
    else if (val === 'false') val = false;
    else if (!isNaN(Number(val)) && val !== '') val = Number(val);
    
    handleFieldChange(path, val);
    setNewFieldName('');
    setNewFieldVal('');
    setNewKeySection(null);
  };

  // Add new section box helper
  const handleAddNewSection = () => {
    if (!newSectionName.trim()) return;
    const secKey = newSectionName.trim().toLowerCase().replace(/\s+/g, '_');
    if (parsedData[secKey]) return;
    const copy = { ...parsedData, [secKey]: {} };
    updateYamlData(copy);
    setNewSectionName('');
    setShowAddSectionModal(false);
  };

  // Toggle collapse state
  const toggleSectionCollapse = (secKey: string) => {
    setCollapsedSections((prev) => ({ ...prev, [secKey]: !prev[secKey] }));
  };

  // Filter sections by search query
  const filteredSections = useMemo(() => {
    const sections = Object.keys(parsedData);
    if (!searchQuery.trim()) return sections;

    const query = searchQuery.toLowerCase();
    return sections.filter((secKey) => {
      if (secKey.toLowerCase().includes(query)) return true;
      const meta = SECTION_META[secKey];
      if (meta && (meta.title.toLowerCase().includes(query) || meta.desc.toLowerCase().includes(query))) return true;

      // Check fields inside
      const secVal = parsedData[secKey];
      if (secVal && typeof secVal === 'object') {
        const jsonString = JSON.stringify(secVal).toLowerCase();
        if (jsonString.includes(query)) return true;
      }
      return false;
    });
  }, [parsedData, searchQuery]);

  // Render Smart Field Controls for Box UI
  const renderFieldControl = (path: string[], value: any) => {
    const keyName = path[path.length - 1];
    const isSensitive = keyName.includes('key') || keyName.includes('secret') || keyName.includes('password') || keyName.includes('token');

    // 1. Boolean Toggle Switch
    if (typeof value === 'boolean') {
      return (
        <div
          className="toggle-switch-container"
          onClick={() => handleFieldChange(path, !value)}
        >
          <div className={`toggle-switch ${value ? 'active' : ''}`}>
            <div className="toggle-switch-handle" />
          </div>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: value ? 'var(--accent-success)' : 'var(--text-muted)' }}>
            {value ? 'ENABLED' : 'DISABLED'}
          </span>
        </div>
      );
    }

    // 2. Dropdown for supported fields (Video codecs, bitrates, FPS, quality profiles, TTS, YouTube, etc.)
    if (DROPDOWN_OPTIONS[keyName]) {
      return (
        <DropdownControl
          value={value}
          options={DROPDOWN_OPTIONS[keyName]}
          onChange={(newVal) => handleFieldChange(path, newVal)}
        />
      );
    }

    // 3. Tag / Array Chips Editor (with Resolution Preset dropdown if key is default_output_resolution)
    if (Array.isArray(value)) {
      if (keyName === 'default_output_resolution') {
        const matchingPreset = RESOLUTION_PRESETS.find(
          (p) => p.value.length === value.length && p.value.every((v, i) => v === value[i])
        );
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Preset:</span>
              <select
                style={{ fontSize: '0.8rem', padding: '6px 10px' }}
                value={matchingPreset ? JSON.stringify(matchingPreset.value) : 'custom'}
                onChange={(e) => {
                  if (e.target.value !== 'custom') {
                    handleFieldChange(path, JSON.parse(e.target.value));
                  }
                }}
              >
                {!matchingPreset && <option value="custom">Custom ({value.join(' x ')})</option>}
                {RESOLUTION_PRESETS.map((p, idx) => (
                  <option key={idx} value={JSON.stringify(p.value)}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
            <TagArrayEditor
              tags={value}
              onChange={(newTags) => handleFieldChange(path, newTags)}
            />
          </div>
        );
      }

      return (
        <TagArrayEditor
          tags={value}
          onChange={(newTags) => handleFieldChange(path, newTags)}
        />
      );
    }

    // 4. Number Input
    if (typeof value === 'number') {
      return (
        <input
          type="number"
          step={Number.isInteger(value) ? '1' : '0.1'}
          value={value}
          onChange={(e) => {
            const num = parseFloat(e.target.value);
            handleFieldChange(path, isNaN(num) ? 0 : num);
          }}
        />
      );
    }

    // 5. Object (Sub-nested fields)
    if (value !== null && typeof value === 'object') {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%', marginTop: '6px' }}>
          {Object.entries(value).map(([subKey, subVal]) => (
            <div
              key={subKey}
              style={{
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px dashed var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 12px',
              }}
            >
              <div className="yaml-field-label" style={{ marginBottom: '6px' }}>
                <span>{formatLabel(subKey)}</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{subKey}</span>
              </div>
              {renderFieldControl([...path, subKey], subVal)}
            </div>
          ))}
        </div>
      );
    }

    // 6. Default String Input
    return (
      <input
        type={isSensitive ? 'password' : 'text'}
        value={value ?? ''}
        placeholder={`Enter ${formatLabel(keyName)}...`}
        onChange={(e) => handleFieldChange(path, e.target.value)}
      />
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Controls Toolbar */}
      <div
        className="glass-card"
        style={{
          padding: '16px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '14px',
        }}
      >
        {/* Left: View Switcher */}
        <div style={{ display: 'flex', gap: '8px', background: 'rgba(0, 0, 0, 0.4)', padding: '4px', borderRadius: 'var(--radius-sm)' }}>
          <button
            className={`btn ${viewMode === 'boxes' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 14px', fontSize: '0.85rem' }}
            onClick={() => setViewMode('boxes')}
          >
            🎛️ Interactive Boxes View
          </button>
          <button
            className={`btn ${viewMode === 'raw' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 14px', fontSize: '0.85rem' }}
            onClick={() => setViewMode('raw')}
          >
            📝 Raw YAML Editor
          </button>
        </div>

        {/* Search Bar (when in Box view mode) */}
        {viewMode === 'boxes' && (
          <div style={{ position: 'relative', flex: 1, maxWidth: '360px' }}>
            <input
              type="text"
              placeholder="🔍 Search configuration settings..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ paddingLeft: '34px', fontSize: '0.85rem' }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                }}
              >
                ✕
              </button>
            )}
          </div>
        )}

        {/* Right Action Controls */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {viewMode === 'boxes' && (
            <>
              <button
                className="btn btn-secondary"
                style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                onClick={() => setShowAddSectionModal(true)}
              >
                ➕ Add Section
              </button>
              <button
                className="btn btn-secondary"
                style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                onClick={() => {
                  const allCollapsed = filteredSections.every((k) => collapsedSections[k]);
                  const nextState: Record<string, boolean> = {};
                  filteredSections.forEach((k) => (nextState[k] = !allCollapsed));
                  setCollapsedSections(nextState);
                }}
              >
                {filteredSections.every((k) => collapsedSections[k]) ? '📂 Expand All' : '📁 Collapse All'}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Parse Error Banner */}
      {parseError && (
        <div
          style={{
            padding: '12px 16px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(255, 71, 87, 0.12)',
            border: '1px solid rgba(255, 71, 87, 0.3)',
            color: 'var(--accent-danger)',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <span>⚠️</span>
          <span style={{ flex: 1 }}>{parseError}</span>
        </div>
      )}

      {/* VIEW MODE 1: INTERACTIVE BOXES */}
      {viewMode === 'boxes' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {filteredSections.length === 0 ? (
            <div className="glass-card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No configuration sections matching "{searchQuery}"
            </div>
          ) : (
            filteredSections.map((sectionKey) => {
              const meta = SECTION_META[sectionKey] || {
                title: formatLabel(sectionKey),
                icon: '⚙️',
                desc: `Custom configuration properties for ${sectionKey}`,
                color: 'accent-api',
              };
              const isCollapsed = collapsedSections[sectionKey];
              const sectionData = parsedData[sectionKey] || {};
              const fieldCount = typeof sectionData === 'object' ? Object.keys(sectionData).length : 1;

              return (
                <div key={sectionKey} className={`yaml-section-box ${meta.color}`}>
                  {/* Section Box Header */}
                  <div
                    className={`yaml-section-header ${isCollapsed ? 'collapsed' : ''}`}
                    onClick={() => toggleSectionCollapse(sectionKey)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span style={{ fontSize: '1.4rem' }}>{meta.icon}</span>
                      <div>
                        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '10px' }}>
                          {meta.title}
                          <span className="badge" style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                            {fieldCount} {fieldCount === 1 ? 'item' : 'items'}
                          </span>
                        </h3>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          {meta.desc}
                        </p>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setNewKeySection(newKeySection === sectionKey ? null : sectionKey);
                        }}
                      >
                        ➕ Field
                      </button>
                      <span style={{ fontSize: '1.1rem', color: 'var(--text-muted)', transition: 'transform 0.2s', transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)' }}>
                        ▼
                      </span>
                    </div>
                  </div>

                  {/* Add New Field Inline Panel */}
                  {newKeySection === sectionKey && (
                    <div
                      style={{
                        margin: '12px 0',
                        padding: '12px 14px',
                        background: 'rgba(108, 92, 231, 0.1)',
                        border: '1px solid rgba(108, 92, 231, 0.3)',
                        borderRadius: 'var(--radius-sm)',
                        display: 'grid',
                        gridTemplateColumns: '180px 1fr 100px 80px',
                        gap: '10px',
                        alignItems: 'center',
                      }}
                    >
                      <input
                        type="text"
                        placeholder="field_name"
                        value={newFieldName}
                        onChange={(e) => setNewFieldName(e.target.value)}
                        style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
                      />
                      <input
                        type="text"
                        placeholder="value (e.g. true, 100, string)"
                        value={newFieldVal}
                        onChange={(e) => setNewFieldVal(e.target.value)}
                        style={{ fontSize: '0.8rem' }}
                      />
                      <button
                        className="btn btn-primary"
                        style={{ padding: '6px', fontSize: '0.8rem', justifyContent: 'center' }}
                        onClick={() => handleAddFieldToSection(sectionKey)}
                      >
                        Add
                      </button>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '6px', fontSize: '0.8rem', justifyContent: 'center' }}
                        onClick={() => setNewKeySection(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  )}

                  {/* Section Content Grid */}
                  {!isCollapsed && (
                    <div className="yaml-field-grid">
                      {typeof sectionData === 'object' && sectionData !== null ? (
                        Object.entries(sectionData).map(([fKey, fVal]) => (
                          <div key={fKey} className="yaml-field-card">
                            <div className="yaml-field-label">
                              <span>{formatLabel(fKey)}</span>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{fKey}</span>
                                <button
                                  className="tag-chip-remove"
                                  title="Delete field"
                                  onClick={() => handleDeleteField([sectionKey, fKey])}
                                >
                                  ×
                                </button>
                              </div>
                            </div>

                            {renderFieldControl([sectionKey, fKey], fVal)}
                          </div>
                        ))
                      ) : (
                        <div className="yaml-field-card">
                          {renderFieldControl([sectionKey], sectionData)}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* VIEW MODE 2: RAW TEXT */}
      {viewMode === 'raw' && (
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ marginBottom: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Edit raw YAML source text directly. Changes auto-sync back to visual cards.
          </div>
          <textarea
            rows={22}
            value={rawText}
            onChange={(e) => handleRawTextChange(e.target.value)}
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.85rem',
              lineHeight: '1.6',
              background: 'rgba(0, 0, 0, 0.4)',
            }}
          />
        </div>
      )}

      {/* Modal: Add New Custom Section */}
      {showAddSectionModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.7)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
          }}
        >
          <div className="glass-card" style={{ width: '400px', padding: '24px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '12px' }}>➕ Create Configuration Section Box</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Enter a name for the new top-level configuration section:
            </p>
            <input
              type="text"
              placeholder="e.g. notifications, caching"
              value={newSectionName}
              onChange={(e) => setNewSectionName(e.target.value)}
              style={{ marginBottom: '20px' }}
              autoFocus
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button className="btn btn-secondary" onClick={() => setShowAddSectionModal(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleAddNewSection} disabled={!newSectionName.trim()}>
                Create Section
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Inline helper component for Array Tag Chips
const TagArrayEditor: React.FC<{ tags: any[]; onChange: (newTags: any[]) => void }> = ({ tags, onChange }) => {
  const [inputVal, setInputVal] = useState('');

  const handleAddTag = () => {
    if (!inputVal.trim()) return;
    let item: any = inputVal.trim();
    if (!isNaN(Number(item))) item = Number(item);
    if (!tags.includes(item)) {
      onChange([...tags, item]);
    }
    setInputVal('');
  };

  const handleRemoveTag = (index: number) => {
    const copy = [...tags];
    copy.splice(index, 1);
    onChange(copy);
  };

  return (
    <div className="tag-chip-container">
      {tags.map((t, idx) => (
        <span key={idx} className="tag-chip">
          {String(t)}
          <button className="tag-chip-remove" onClick={() => handleRemoveTag(idx)}>
            ×
          </button>
        </span>
      ))}
      <input
        type="text"
        className="tag-input-inline"
        placeholder="+ tag (press Enter)"
        value={inputVal}
        onChange={(e) => setInputVal(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            handleAddTag();
          }
        }}
      />
    </div>
  );
};
