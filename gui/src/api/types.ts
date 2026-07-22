export interface Project {
  name: string;
  current_phase: string;
  status: string;
  created_at: string;
  updated_at: string;
  has_script: boolean;
  has_video: boolean;
  transition_count: number;
  reddit_url?: string;
}

export interface ScriptSegment {
  time_seconds: number;
  intended_duration_seconds: number;
  narration: string;
  expression_cue?: string;
  visual_asset_path?: string;
  visual_directive?: string;
  text_overlay?: string;
  evidence_refs: string[];
  pace: 'fast' | 'normal' | 'slow';
  emotion: 'excited' | 'calm' | 'dramatic' | 'neutral';
}

export interface ScriptData {
  phase: string;
  title: string;
  hook: string;
  loop_bridge?: string;
  segments: ScriptSegment[];
  sources_to_check: string[];
  hashtags: string[];
}

export interface ArtifactFile {
  name: string;
  folder: string;
  relative_path: string;
  byte_size: number;
  updated_at: number;
}

export interface MediaAsset {
  name: string;
  type: 'image' | 'video';
  category: string;
  url: string;
  byte_size: number;
}

export interface ConfigData {
  yaml_content: string;
  env_vars: Record<string, string>;
}

export interface PreflightCheckItem {
  name: string;
  passed: boolean;
  message: string;
}

export interface PreflightResult {
  all_passed: boolean;
  checks: PreflightCheckItem[];
}

export interface WSLogMessage {
  type: 'log';
  project: string;
  message: string;
  level: 'info' | 'warning' | 'error';
  phase?: string;
}

export interface WSEventMessage {
  type: string;
  project: string;
  data: Record<string, any>;
}
