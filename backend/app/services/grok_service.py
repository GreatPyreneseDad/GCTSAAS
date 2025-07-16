"""
GitHub Models Grok 3 Integration Service
"""
import httpx
import json
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GrokService:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.base_url = "https://models.github.com/v1/chat/completions"
        
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable required")
    
    async def generate_assessment_scenarios(self, 
                                          variable_type: str,
                                          cultural_context: str = "western_corporate",
                                          count: int = 5) -> List[Dict]:
        """
        Generate culturally adapted assessment scenarios using Grok 3
        """
        prompt = self._build_scenario_prompt(variable_type, cultural_context, count)
        
        try:
            response = await self._call_grok3(prompt)
            scenarios = json.loads(response)
            return scenarios
        except Exception as e:
            logger.error(f"Failed to generate scenarios: {e}")
            return self._get_fallback_scenarios(variable_type)
    
    async def analyze_coherence_pattern(self, 
                                      coherence_history: List[Dict],
                                      user_context: Dict) -> Dict:
        """
        Analyze coherence patterns and provide insights
        """
        prompt = self._build_analysis_prompt(coherence_history, user_context)
        
        try:
            response = await self._call_grok3(prompt)
            analysis = json.loads(response)
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze pattern: {e}")
            return {"error": "Analysis unavailable", "fallback": True}
    
    def _build_scenario_prompt(self, variable_type: str, context: str, count: int) -> str:
        variable_definitions = {
            "psi": "Internal consistency between beliefs, actions, and emotions across contexts",
            "rho": "Accumulated wisdom from survived difficulties and integrated learning",
            "q": "Moral activation energy when core values are violated",
            "f": "Quality and consistency of meaningful social connections"
        }
        
        definition = variable_definitions.get(variable_type, "Unknown variable")
        
        return f"""
        Generate {count} assessment scenarios for measuring {variable_type} ({definition}) 
        in {context} cultural context.

        Each scenario must:
        1. Test the specific coherence variable according to GCT principles
        2. Be culturally appropriate and relevant
        3. Allow quantitative scoring (0.0-1.0 scale)
        4. Include clear measurement rubric

        Return as JSON array with format:
        [{{
            "scenario": "description",
            "measurement_method": "how to score",
            "scoring_rubric": "detailed criteria",
            "cultural_notes": "adaptation considerations"
        }}]

        Respond ONLY with valid JSON, no additional text.
        """
    
    def _build_analysis_prompt(self, history: List[Dict], context: Dict) -> str:
        return f"""
        Analyze this coherence trajectory data according to GCT principles:
        
        History: {json.dumps(history[-10:])}  # Last 10 measurements
        Context: {json.dumps(context)}
        
        Provide analysis in JSON format:
        {{
            "trend_analysis": "overall trajectory description",
            "inflection_points": ["list of significant changes"],
            "component_drivers": {{"which variables driving changes"}},
            "predictions": "likely future trajectory",
            "recommendations": ["actionable insights"],
            "confidence_level": 0.0-1.0
        }}
        
        Respond ONLY with valid JSON.
        """
    
    async def _call_grok3(self, prompt: str) -> str:
        """Make API call to Grok 3 via GitHub Models"""
        headers = {
            "Authorization": f"token {self.github_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "grok-3",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert in Grounded Coherence Theory and organizational psychology. Provide precise, actionable responses in valid JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
    
    def _get_fallback_scenarios(self, variable_type: str) -> List[Dict]:
        """Fallback scenarios if Grok 3 unavailable"""
        fallback_scenarios = {
            "psi": [
                {
                    "scenario": "Your team is facing a critical deadline. Your stated value is 'work-life balance' but you're asked to work overtime.",
                    "measurement_method": "Response consistency with stated values",
                    "scoring_rubric": "1.0 = Response fully aligns with stated values, 0.0 = Complete contradiction",
                    "cultural_notes": "Western corporate context"
                }
            ],
            "rho": [
                {
                    "scenario": "Describe a significant challenge you overcame and what you learned.",
                    "measurement_method": "Quality of learning integration",
                    "scoring_rubric": "1.0 = Deep insight with applied learning, 0.0 = Surface-level or no learning",
                    "cultural_notes": "Universal applicability"
                }
            ]
        }
        
        return fallback_scenarios.get(variable_type, [])
