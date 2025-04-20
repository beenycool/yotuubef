# Plan to Fix and Improve Reddit-to-YouTube Script (v2)

This plan addresses issues identified in `script.py` and incorporates Gemini 2.0 Flash multimodal analysis, smarter trimming, ElevenLabs title voiceover, and on-screen text display.

**1. Update Gemini Integration (Multimodal Analysis):**

*   **Goal:** Leverage Gemini 2.0 Flash for video understanding (hashtags, trimming, subtitle needs & content).
*   **Action 1:** Update model ID to `gemini-2.0-flash`.
*   **Action 2:** Implement multimodal API calls (sending video data/frames). Consult Gemini documentation for exact format.
*   **Action 3:** Modify the Gemini prompt to request:
    *   Relevant hashtags based on video content.
    *   The best 60-second segment (start/end times) if the video is > 60s.
    *   A determination of whether transcription/subtitles are necessary (e.g., if dialogue is present).
    *   If subtitles *are* necessary, provide the subtitle text.

**2. Integrate ElevenLabs API (Title Voiceover Only):**

*   **Goal:** Generate audio reading the video title.
*   **Action 1:** Add ElevenLabs API key handling. **Decision:** Hardcode the key directly in the script as per user instruction (Note: Not best practice for security).
*   **Action 2:** Add function `synthesize_title_audio(title, output_audio_path)` that calls the ElevenLabs API with the Reddit post title.

**3. Refine Video Processing Workflow & Editing (`moviepy`):**

*   Download Video.
*   Get Full Duration & Dimensions.
*   **Analyze with Gemini (Multimodal):** Get hashtags, best 60s segment, subtitle need flag, and subtitle text (if needed).
*   **Trim Video:** Based on Gemini segment or first 60s fallback.
*   Check Aspect Ratio. Determine `is_short`.
*   **Synthesize Title Audio (ElevenLabs).**
*   **Create Final Video (`moviepy`):**
    *   Take the trimmed video clip as the base.
    *   **Title Sequence:**
        *   Create a `TextClip` for the Reddit title (fit to screen, styled).
        *   Determine duration (match audio or fixed).
        *   Overlay `TextClip` at the start.
        *   Set synthesized title audio to play during this sequence.
    *   **Subtitle Sequence:**
        *   If Gemini indicated subtitles are needed:
            *   Create `TextClip`(s) for subtitle text.
            *   Position appropriately (e.g., bottom/middle center).
            *   Overlay subtitles *after* title sequence.
    *   Composite all elements.
*   Upload to YouTube.
*   Cleanup Files.

**4. Update Dependencies & Configuration:**

*   Add/Update libraries: `google-generativeai`, `elevenlabs`, `moviepy`, potentially `opencv-python`.
*   Handle API keys: Gemini (via existing variable/method), ElevenLabs (hardcoded).

**Key Considerations:**

*   **Gemini Subtitle Output Format:** Need to adapt parsing based on how Gemini returns subtitles.
*   **Text Styling/Fitting (`moviepy`):** Requires careful implementation for dynamic text.
*   **Timing (`moviepy`):** Precise control needed for synchronizing audio and text overlays.

**5. Next Steps:**

1.  Plan approved and saved to `PLAN.md`.
2.  Switch to "Code" mode to implement the changes in `script.py`.