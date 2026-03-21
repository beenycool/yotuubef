import ast
import re

with open("src/enhanced_orchestrator.py", "r") as f:
    content = f.read()

# We will replace `process_enhanced_video` logic with sub-functions.

def replace_between(content, start_str, end_str, new_str):
    start = content.find(start_str)
    if start == -1:
        return content
    end = content.find(end_str, start)
    if end == -1:
        return content
    return content[:start] + new_str + content[end:]

old_method_start = """    async def process_enhanced_video(
        self, reddit_url: str, enhanced_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        \"\"\"Run faceless lore pipeline: text post -> research -> script -> TTS -> Minecraft bg.\"\"\"
        try:"""

new_method = """    async def process_enhanced_video(
        self, reddit_url: str, enhanced_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        \"\"\"Run faceless lore pipeline: text post -> research -> script -> TTS -> Minecraft bg.\"\"\"
        try:
            self.logger.info("Starting faceless lore generation for: %s", reddit_url)

            reddit_post = await self._fetch_and_validate_reddit_post(reddit_url)
            if isinstance(reddit_post, dict):
                return reddit_post

            analysis = await self._generate_script_and_research(reddit_post)
            if not analysis:
                return {"success": False, "error": "Script generation failed"}

            tts_paths = self._generate_tts_audio(analysis)
            if not tts_paths:
                return {"success": False, "error": "TTS generation failed"}

            audio_segments, audio_clip, final_video, output_file = self._create_final_video(reddit_post, analysis, tts_paths)

            upload_result = await self.youtube_client.upload_video(
                str(output_file), {
                    "title": analysis.suggested_title,
                    "description": analysis.summary_for_description,
                    "tags": [tag.replace("#", "") for tag in analysis.hashtags],
                }
            )

            self._cleanup_resources(audio_segments, audio_clip, final_video)

            if not upload_result.get("success"):
                return {
                    "success": False,
                    "error": upload_result.get("error", "Upload failed"),
                    "video_path": str(output_file),
                }

            return {
                "success": True,
                "video_id": upload_result.get("video_id"),
                "video_url": upload_result.get("video_url"),
                "video_path": str(output_file),
                "pipeline": "faceless_lore",
            }

        except Exception as e:
            self.logger.error("Lore generation failed: %s", e, exc_info=True)
            self.gpu_manager.clear_gpu_cache()
            return {"success": False, "error": str(e), "stage": "faceless_lore"}

    async def _fetch_and_validate_reddit_post(self, reddit_url: str) -> Any:
        async with RedditClient() as reddit_client:
            reddit_post = await reddit_client.get_post_by_url(reddit_url)

        if not reddit_post:
            return {"success": False, "error": "Failed to load Reddit post"}
        if reddit_post.is_video:
            return {
                "success": False,
                "error": "Provided URL points to a video post. Faceless flow requires text posts.",
            }
        if (
            not getattr(reddit_post, "selftext", "")
            or len(reddit_post.selftext.strip()) < 120
        ):
            return {
                "success": False,
                "error": "Text post is too short for lore generation.",
            }
        return reddit_post

    async def _generate_script_and_research(self, reddit_post: Any) -> Any:
        researcher = DeepResearchClient()
        query = f"{reddit_post.title} {reddit_post.subreddit} history"
        research_facts = await researcher.conduct_deep_research(query)

        reddit_content_dict = {
            "title": reddit_post.title,
            "selftext": reddit_post.selftext,
            "subreddit": reddit_post.subreddit,
            "score": reddit_post.score,
            "num_comments": reddit_post.num_comments,
            "deep_research": research_facts,
        }
        return await self.ai_client.analyze_video_content(
            None, reddit_content_dict
        )

    def _generate_tts_audio(self, analysis: Any) -> list:
        tts_results = (
            self.advanced_audio_processor.tts_service.generate_multiple_segments(
                analysis.narrative_script_segments
            )
        )
        return [
            item.get("audio_path")
            for item in tts_results
            if item.get("success") and item.get("audio_path")
        ]

    def _create_final_video(self, reddit_post: Any, analysis: Any, tts_paths: list) -> tuple:
        audio_segments = [AudioFileClip(str(path)) for path in tts_paths]
        audio_clip = concatenate_audioclips(audio_segments)

        bg_manager = BackgroundManager()
        video_clip = bg_manager.get_sliced_background(
            target_duration=audio_clip.duration,
            subreddit=reddit_post.subreddit,
            text_content=reddit_post.selftext,
        )
        if analysis.text_overlays:
            video_clip = self.video_processor.text_processor.add_text_overlays(
                video_clip, analysis.text_overlays
            )
        final_video = MoviePyCompat.with_audio(video_clip, audio_clip)

        output_file = self.config.paths.processed_dir / f"lore_{reddit_post.id}.mp4"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_file),
            fps=30,
            codec="libx264",
            audio_codec="aac",
        )
        return audio_segments, audio_clip, final_video, output_file

    def _cleanup_resources(self, audio_segments: list, audio_clip: Any, final_video: Any) -> None:
        for segment in audio_segments:
            try:
                segment.close()
            except Exception:
                pass
        try:
            audio_clip.close()
        except Exception:
            pass
        try:
            final_video.close()
        except Exception:
            pass"""

content = replace_between(content, old_method_start, "    async def process_ai_production_studio(", new_method + "\n\n")

with open("src/enhanced_orchestrator.py", "w") as f:
    f.write(content)
