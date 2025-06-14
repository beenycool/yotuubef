"""
Strategic Narrative Analysis Module
Implements the "narrative gap" approach for content curation and storytelling.
This module analyzes Reddit content for narrative potential and creates compelling stories.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import get_config
from src.models import RedditPost, NarrativeGap, NarrativeAnalysis, CharacterProfile


class NarrativeAnalyzer:
    """
    Analyzes Reddit content for narrative gaps and storytelling potential.
    Implements the strategic approach of finding content with unexplained elements
    that can be filled with compelling narration.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Narrative gap patterns
        self.gap_patterns = {
            "context_missing": [
                r"\b(what|why|how|when|where)\b",
                r"\b(suddenly|unexpectedly|out of nowhere)\b",
                r"\b(mysterious|unknown|unclear|confusing)\b"
            ],
            "reaction_unexplained": [
                r"\b(look|stare|expression|face|reaction)\b",
                r"\b(shocked|surprised|confused|amazed)\b",
                r"\b(wtf|what the|omg|wow)\b"
            ],
            "outcome_unclear": [
                r"\b(what happens|what happened|then what|but then)\b",
                r"\b(cliffhanger|to be continued|wait for it)\b",
                r"\b(you won't believe|never saw coming)\b"
            ],
            "character_mystery": [
                r"\b(this (guy|girl|person|animal))\b",
                r"\b(who is|what is|meet)\b",
                r"\b(character|personality|behavior)\b"
            ]
        }
        
        # Character detection patterns
        self.character_patterns = {
            "person": [
                r"\b(man|woman|guy|girl|person|dude|lady)\b",
                r"\b(he|she|they|someone|this person)\b"
            ],
            "animal": [
                r"\b(dog|cat|bird|owl|rabbit|horse|cow|pig|chicken|duck|fish|snake|lion|tiger|bear|elephant|monkey|dolphin|whale|shark|turtle|frog|spider|bee|butterfly|ant|fly|mouse|rat|hamster|guinea pig|ferret|lizard|gecko|iguana|chameleon|parrot|canary|peacock|flamingo|penguin|ostrich|eagle|hawk|owl|crow|raven|robin|sparrow|pigeon|seagull|pelican|heron|swan|goose|duck|turkey|chicken|rooster|hen|goat|sheep|lamb|cow|bull|pig|boar|deer|elk|moose|buffalo|bison|zebra|giraffe|hippopotamus|rhinoceros|crocodile|alligator|komodo dragon|kangaroo|koala|panda|polar bear|grizzly bear|black bear|wolf|fox|coyote|jackal|hyena|leopard|cheetah|jaguar|cougar|lynx|bobcat|ocelot|serval|caracal|sand cat|pallas cat|scottish fold|persian cat|siamese cat|maine coon|ragdoll|british shorthair|russian blue|bengal cat|abyssinian|american shorthair|exotic shorthair|siberian cat|norwegian forest cat|birman|burmese|oriental shorthair|devon rex|cornish rex|sphynx|munchkin|scottish straight|american curl|japanese bobtail|turkish angora|turkish van|manx|cymric|laperm|selkirk rex|american wirehair|havana brown|singapura|tonkinese|ocicat|egyptian mau|korat|chartreux|nebelung|pixie bob|american bobtail|highlander|chausie|savannah|serengeti|toyger|cheetoh|safari|bristol|bramble|desert lynx|jungle cat|sand cat|black footed cat|rusty spotted cat|flat headed cat|fishing cat|leopard cat|pallas cat|manul|chinese mountain cat|european wildcat|african wildcat|jungle cat|sand cat|black footed cat|rusty spotted cat|flat headed cat|fishing cat|leopard cat|pallas cat|manul|chinese mountain cat|european wildcat|african wildcat)\b",
                r"\b(pet|animal|creature|beast|wild|domestic)\b"
            ],
            "object": [
                r"\b(thing|object|item|device|machine|tool|gadget)\b",
                r"\b(it|this|that|something)\b"
            ]
        }
        
        # Narrator personas and their characteristics
        self.narrator_personas = {
            "documentary": {
                "description": "David Attenborough-style nature documentary narrator",
                "tone": "educational, authoritative, slightly dramatic",
                "example_phrases": ["In the wild world of...", "Observe as...", "Here we witness...", "Nature's most fascinating..."]
            },
            "comedic": {
                "description": "Humorous, lighthearted commentary",
                "tone": "funny, relatable, casual",
                "example_phrases": ["This is the moment when...", "You know that feeling when...", "Plot twist:", "Meanwhile, in..."]
            },
            "curious": {
                "description": "Inquisitive, mystery-solving narrator",
                "tone": "questioning, investigative, engaging",
                "example_phrases": ["But wait, there's more...", "The real question is...", "What you're seeing here is...", "The mystery deepens..."]
            },
            "whimsical": {
                "description": "Playful, imaginative storyteller",
                "tone": "magical, creative, entertaining",
                "example_phrases": ["Once upon a time...", "In a world where...", "Little did they know...", "The adventure begins..."]
            }
        }
    
    def analyze_narrative_potential(self, post: RedditPost) -> NarrativeAnalysis:
        """
        Analyze a Reddit post for narrative potential and storytelling opportunities.
        
        Args:
            post: RedditPost object to analyze
            
        Returns:
            NarrativeAnalysis with complete narrative assessment
        """
        try:
            self.logger.info(f"Analyzing narrative potential for post: {post.id}")
            
            # Identify narrative gaps
            narrative_gaps = self._identify_narrative_gaps(post)
            
            # Analyze character/subject
            character_profile = self._analyze_character(post)
            
            # Determine story arc
            story_arc = self._determine_story_arc(post, narrative_gaps, character_profile)
            
            # Calculate narrative potential score
            narrative_score = self._calculate_narrative_score(post, narrative_gaps, character_profile)
            
            # Generate Hook-Story-Payoff structure
            hook_story_payoff = self._generate_hook_story_payoff(post, narrative_gaps, character_profile, story_arc)
            
            # Select optimal narrator persona
            narrator_persona = self._select_narrator_persona(post, character_profile, story_arc)
            
            # Estimate retention potential
            retention_score = self._estimate_retention(narrative_score, len(narrative_gaps), character_profile)
            
            analysis = NarrativeAnalysis(
                post_id=post.id,
                narrative_gaps=narrative_gaps,
                character_profile=character_profile,
                story_arc=story_arc,
                narrative_potential_score=narrative_score,
                hook_story_payoff=hook_story_payoff,
                narrator_persona=narrator_persona,
                estimated_retention=retention_score
            )
            
            self.logger.info(f"Narrative analysis complete. Score: {narrative_score}/100, Gaps: {len(narrative_gaps)}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Narrative analysis failed for post {post.id}: {e}")
            # Return minimal analysis on failure
            return NarrativeAnalysis(
                post_id=post.id,
                narrative_gaps=[],
                character_profile=None,
                story_arc="unknown",
                narrative_potential_score=0,
                hook_story_payoff={"hook": "", "story": "", "payoff": ""},
                narrator_persona="curious",
                estimated_retention=0
            )
    
    def _identify_narrative_gaps(self, post: RedditPost) -> List[NarrativeGap]:
        """Identify narrative gaps in the post content"""
        gaps = []
        text_to_analyze = f"{post.title} {post.subreddit}".lower()
        
        for gap_type, patterns in self.gap_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_to_analyze):
                    gap = self._create_narrative_gap(gap_type, post, pattern)
                    if gap:
                        gaps.append(gap)
                    break  # Only one gap per type
        
        # Special case: Video with no clear context (high narrative potential)
        if post.is_video and len(post.title.split()) <= 3:
            gaps.append(NarrativeGap(
                gap_type="context_missing",
                description="Video has minimal context, perfect for storytelling",
                fill_strategy="Provide narrative context and interpretation",
                engagement_potential=85,
                example_hook="You're about to witness something extraordinary..."
            ))
        
        return gaps
    
    def _create_narrative_gap(self, gap_type: str, post: RedditPost, matched_pattern: str) -> Optional[NarrativeGap]:
        """Create a specific narrative gap based on type and context"""
        gap_configs = {
            "context_missing": {
                "description": "Situation lacks clear context or explanation",
                "fill_strategy": "Provide background story and context",
                "engagement_potential": 75,
                "example_hook": "What you're seeing here is more than meets the eye..."
            },
            "reaction_unexplained": {
                "description": "Character reaction is visible but unexplained",
                "fill_strategy": "Interpret and explain the emotional response",
                "engagement_potential": 85,
                "example_hook": "That look says everything you need to know..."
            },
            "outcome_unclear": {
                "description": "Result or conclusion is ambiguous",
                "fill_strategy": "Provide narrative resolution or interpretation",
                "engagement_potential": 70,
                "example_hook": "The ending will surprise you..."
            },
            "character_mystery": {
                "description": "Main character/subject has unclear motivations",
                "fill_strategy": "Give character depth and personality",
                "engagement_potential": 80,
                "example_hook": "Meet the star of today's story..."
            }
        }
        
        config = gap_configs.get(gap_type)
        if not config:
            return None
            
        return NarrativeGap(
            gap_type=gap_type,
            description=config["description"],
            fill_strategy=config["fill_strategy"],
            engagement_potential=config["engagement_potential"],
            example_hook=config["example_hook"]
        )
    
    def _analyze_character(self, post: RedditPost) -> Optional[CharacterProfile]:
        """Analyze and profile the main character/subject"""
        text_to_analyze = f"{post.title} {post.subreddit}".lower()
        
        # Determine character type
        character_type = "situation"  # Default
        for char_type, patterns in self.character_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_to_analyze):
                    character_type = char_type
                    break
            if character_type != "situation":
                break
        
        # Generate character profile based on type and context
        if character_type == "animal":
            return CharacterProfile(
                type="animal",
                description="Animal subject with expressive behavior",
                personality_traits=["expressive", "unpredictable", "authentic"],
                emotional_state="curious",
                narrative_role="protagonist"
            )
        elif character_type == "person":
            return CharacterProfile(
                type="person",
                description="Human subject in interesting situation",
                personality_traits=["relatable", "genuine", "engaging"],
                emotional_state="surprised",
                narrative_role="protagonist"
            )
        elif character_type == "object":
            return CharacterProfile(
                type="object",
                description="Inanimate object with unexpected behavior",
                personality_traits=["mysterious", "surprising", "intriguing"],
                emotional_state="neutral",
                narrative_role="mystery"
            )
        else:
            return CharacterProfile(
                type="situation",
                description="Interesting situation or scenario",
                personality_traits=["unexpected", "engaging", "relatable"],
                emotional_state="dynamic",
                narrative_role="context"
            )
    
    def _determine_story_arc(self, post: RedditPost, gaps: List[NarrativeGap], character: Optional[CharacterProfile]) -> str:
        """Determine the best story arc for the content"""
        text_to_analyze = f"{post.title} {post.subreddit}".lower()
        
        # Comedy indicators
        if any(word in text_to_analyze for word in ["funny", "hilarious", "comedy", "joke", "meme"]) or \
           post.subreddit.lower() in ["funny", "memes", "comedy", "humor"]:
            return "comedy"
        
        # Mystery indicators
        if any(gap.gap_type in ["context_missing", "character_mystery"] for gap in gaps) or \
           any(word in text_to_analyze for word in ["mystery", "strange", "weird", "unexplained"]):
            return "mystery"
        
        # Drama indicators
        if any(word in text_to_analyze for word in ["dramatic", "intense", "emotional", "touching"]):
            return "drama"
        
        # Transformation indicators
        if any(word in text_to_analyze for word in ["before", "after", "change", "transform", "becomes"]):
            return "transformation"
        
        # Revelation indicators
        if any(word in text_to_analyze for word in ["reveal", "discover", "found", "secret", "hidden"]):
            return "revelation"
        
        # Default based on character type
        if character and character.type == "animal":
            return "comedy"
        elif character and character.type == "person":
            return "mystery"
        else:
            return "mystery"
    
    def _calculate_narrative_score(self, post: RedditPost, gaps: List[NarrativeGap], character: Optional[CharacterProfile]) -> int:
        """Calculate overall narrative potential score"""
        score = 0
        
        # Base score from engagement metrics
        score += min(30, post.score // 10)  # Up to 30 points from Reddit score
        score += min(20, post.num_comments // 5)  # Up to 20 points from comments
        
        # Narrative gaps contribution
        if gaps:
            avg_gap_potential = sum(gap.engagement_potential for gap in gaps) / len(gaps)
            score += int(avg_gap_potential * 0.3)  # Up to 30 points from gaps
        
        # Character presence
        if character:
            if character.type in ["animal", "person"]:
                score += 15  # Clear character adds engagement
            else:
                score += 5
        
        # Video length bonus (optimal for Shorts)
        if post.is_video and post.duration:
            if 5 <= post.duration <= 60:
                score += 10
            elif post.duration < 5:
                score -= 5
        
        # Subreddit bonus for narrative-friendly communities
        narrative_friendly_subs = [
            "unexpected", "wholesomegifs", "animalsbeingderps", "instant_regret",
            "watchpeopledieinside", "oddlysatisfying", "interestingasfuck",
            "nextfuckinglevel", "mildlyinteresting", "blackmagicfuckery"
        ]
        
        if post.subreddit.lower() in narrative_friendly_subs:
            score += 10
        
        return min(100, max(0, score))
    
    def _generate_hook_story_payoff(self, post: RedditPost, gaps: List[NarrativeGap], 
                                  character: Optional[CharacterProfile], story_arc: str) -> Dict[str, str]:
        """Generate Hook-Story-Payoff structure for the content"""
        
        # Select primary narrative gap for structure
        primary_gap = gaps[0] if gaps else None
        
        # Generate hook based on gap and character
        if primary_gap and primary_gap.gap_type == "reaction_unexplained":
            hook = f"This {character.type if character else 'moment'} is about to have a priceless reaction."
        elif primary_gap and primary_gap.gap_type == "context_missing":
            hook = f"What you're seeing here is more than meets the eye."
        elif character and character.type == "animal":
            hook = f"This {character.type} thought it was just another ordinary day."
        else:
            hook = f"You're about to witness something extraordinary."
        
        # Generate story (middle narration)
        if story_arc == "comedy":
            story = f"Watch as our {character.type if character else 'subject'} discovers that life has other plans."
        elif story_arc == "mystery":
            story = f"The {character.type if character else 'situation'} reveals more than expected."
        elif story_arc == "drama":
            story = f"This moment captures the essence of {character.emotional_state if character else 'surprise'}."
        else:
            story = f"What happens next will change everything."
        
        # Generate payoff (conclusion)
        if primary_gap and primary_gap.gap_type == "reaction_unexplained":
            payoff = f"That expression says it all."
        elif story_arc == "comedy":
            payoff = f"Sometimes the best moments are unplanned."
        elif story_arc == "mystery":
            payoff = f"The mystery is solved, but the wonder remains."
        else:
            payoff = f"And that's how you create an unforgettable moment."
        
        return {
            "hook": hook,
            "story": story,
            "payoff": payoff
        }
    
    def _select_narrator_persona(self, post: RedditPost, character: Optional[CharacterProfile], story_arc: str) -> str:
        """Select the optimal narrator persona for the content"""
        
        # Animal content often works well with documentary style
        if character and character.type == "animal":
            return "documentary"
        
        # Comedy arc suits comedic narrator
        if story_arc == "comedy":
            return "comedic"
        
        # Mystery/revelation works with curious narrator
        if story_arc in ["mystery", "revelation"]:
            return "curious"
        
        # Drama and transformation work with whimsical
        if story_arc in ["drama", "transformation"]:
            return "whimsical"
        
        # Default to curious for engagement
        return "curious"
    
    def _estimate_retention(self, narrative_score: int, gap_count: int, character: Optional[CharacterProfile]) -> int:
        """Estimate viewer retention based on narrative elements"""
        base_retention = 60  # Base retention for Shorts
        
        # Narrative score contribution
        base_retention += (narrative_score - 50) * 0.4  # Scale narrative score impact
        
        # Narrative gaps boost retention
        base_retention += gap_count * 5
        
        # Character presence boosts retention
        if character:
            if character.type in ["animal", "person"]:
                base_retention += 10
            else:
                base_retention += 5
        
        return min(95, max(30, int(base_retention)))
    
    def get_enhanced_prompts_for_analysis(self, analysis: NarrativeAnalysis, post: RedditPost) -> Dict[str, str]:
        """
        Generate enhanced prompts for AI analysis based on narrative insights.
        
        Args:
            analysis: NarrativeAnalysis results
            post: Original RedditPost
            
        Returns:
            Dict with enhanced prompts for different AI tasks
        """
        persona_config = self.narrator_personas.get(analysis.narrator_persona, self.narrator_personas["curious"])
        
        # Enhanced TTS generation prompt
        tts_prompt = f"""
        Create a compelling {analysis.narrator_persona} narration for this {post.duration:.1f}s video with a {analysis.story_arc} arc.
        
        Narrative Structure:
        - Hook (0-3s): {analysis.hook_story_payoff['hook']}
        - Story (3-15s): {analysis.hook_story_payoff['story']}
        - Payoff (15s+): {analysis.hook_story_payoff['payoff']}
        
        Persona: {persona_config['description']}
        Tone: {persona_config['tone']}
        
        Character: {analysis.character_profile.description if analysis.character_profile else 'Situation-based content'}
        
        Narrative Gaps to Address:
        {chr(10).join(f"- {gap.description}: {gap.fill_strategy}" for gap in analysis.narrative_gaps)}
        
        Generate a script that fills these narrative gaps while maintaining the {analysis.narrator_persona} persona.
        Keep total duration under {post.duration:.1f} seconds.
        """
        
        # Enhanced visual analysis prompt
        visual_prompt = f"""
        Analyze this video for cinematic editing opportunities based on narrative analysis:
        
        Story Arc: {analysis.story_arc}
        Character Type: {analysis.character_profile.type if analysis.character_profile else 'situational'}
        Key Narrative Moments to Emphasize:
        {chr(10).join(f"- {gap.description}" for gap in analysis.narrative_gaps)}
        
        Focus on:
        1. Visual moments that support the narrative gaps
        2. Character expressions and reactions
        3. Timing for maximum narrative impact
        4. Zoom and speed effects that enhance storytelling
        
        Provide specific timestamps and editing suggestions.
        """
        
        # Enhanced thumbnail prompt
        thumbnail_prompt = f"""
        Create thumbnail concepts that leverage the narrative gaps:
        
        Primary Gap: {analysis.narrative_gaps[0].description if analysis.narrative_gaps else 'Visual intrigue'}
        Character: {analysis.character_profile.type if analysis.character_profile else 'Situational'}
        Story Arc: {analysis.story_arc}
        
        Thumbnail should create curiosity about:
        {chr(10).join(f"- {gap.description}" for gap in analysis.narrative_gaps[:2])}
        
        Suggest optimal frame timing and text overlay that hints at the narrative mystery.
        """
        
        return {
            "tts_prompt": tts_prompt,
            "visual_prompt": visual_prompt,
            "thumbnail_prompt": thumbnail_prompt,
            "title_prompt": f"Create a title that hints at the narrative gap: {analysis.narrative_gaps[0].description if analysis.narrative_gaps else 'intriguing content'}"
        }