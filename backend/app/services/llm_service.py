import os
import json
import time
import requests
from typing import Dict, Any, Optional
from app.config import settings
from app.utils.exceptions import LLMServiceError
from app.utils.logger import logger


class LLMService:
    """Service for LLM integration (Gemini, Claude, etc.)"""
    
    def __init__(self):
        self.gemini_api_key = settings.gemini_api_key
        self.timeout = 30  # seconds
    
    def generate_farmer_report(self, disease_name: str, symptoms: list, 
                             treatments: list, confidence_score: float,
                             additional_context: Dict[str, Any] = None) -> str:
        """
        Generate a farmer-friendly diagnostic report using LLM
        
        Args:
            disease_name: Predicted disease name
            symptoms: List of observed symptoms
            treatments: List of recommended treatments
            confidence_score: Model confidence score (0-1)
            additional_context: Additional context like weather, location, etc.
            
        Returns:
            Generated report as string
        """
        
        prompt = self._build_farmer_prompt(
            disease_name, symptoms, treatments, confidence_score, additional_context
        )
        
        if self.gemini_api_key:
            try:
                return self._call_gemini_api(prompt)
            except Exception as e:
                logger.error(f"Gemini API failed: {str(e)}")
                return self._generate_fallback_report(disease_name, symptoms, treatments, confidence_score)
        else:
            logger.warning("No LLM API key configured, using fallback report")
            return self._generate_fallback_report(disease_name, symptoms, treatments, confidence_score)
    
    def _build_farmer_prompt(self, disease_name: str, symptoms: list, 
                           treatments: list, confidence_score: float,
                           additional_context: Dict[str, Any] = None) -> str:
        """Build prompt for LLM"""
        
        confidence_percent = confidence_score * 100
        
        prompt = f"""
Act as an experienced agricultural expert specializing in mango cultivation. 
Based on the following AI analysis of a mango leaf, provide a clear, practical report suitable for farmers.

DIAGNOSIS INFORMATION:
- Disease Identified: {disease_name}
- Confidence Level: {confidence_percent:.1f}%
- Observed Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
- Recommended Treatments: {', '.join(treatments) if treatments else 'Not specified'}

"""

        if additional_context:
            prompt += f"ADDITIONAL CONTEXT:\n"
            for key, value in additional_context.items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
            prompt += "\n"

        prompt += """
REQUIREMENTS:
1. Use simple, non-technical language that farmers can easily understand
2. Structure the report with clear headings
3. Include practical, actionable steps
4. Add preventive measures for future protection
5. Mention when to seek professional help
6. Include warnings about potential crop impact
7. Consider cost-effectiveness of treatments
8. Provide timeline expectations for recovery

Format the response in a friendly, encouraging tone while being informative and accurate.
"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API for report generation"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite:generateContent?key={self.gemini_api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        start_time = time.time()
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                text_response = result['candidates'][0]['content']['parts'][0]['text']
                generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                logger.info(f"LLM report generated successfully in {generation_time:.0f}ms")
                return text_response.strip()
            else:
                raise LLMServiceError("Invalid response format from Gemini API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {str(e)}")
            raise LLMServiceError(f"Failed to call Gemini API: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini API response: {str(e)}")
            raise LLMServiceError(f"Invalid JSON response from Gemini API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Gemini API call: {str(e)}")
            raise LLMServiceError(f"Unexpected error: {str(e)}")
    
    def _generate_fallback_report(self, disease_name: str, symptoms: list, 
                                treatments: list, confidence_score: float) -> str:
        """Generate a fallback report when LLM is unavailable"""
        
        confidence_percent = confidence_score * 100
        
        report = f"""
🥭 MANGO LEAF DISEASE REPORT
{'='*40}

DIAGNOSIS:
• Disease: {disease_name}
• Confidence: {confidence_percent:.1f}%

SYMPTOMS IDENTIFIED:
"""
        
        for i, symptom in enumerate(symptoms, 1):
            report += f"• {symptom}\n"
        
        report += f"""
RECOMMENDED TREATMENTS:
"""
        
        for i, treatment in enumerate(treatments, 1):
            report += f"• {treatment}\n"
        
        report += f"""
IMPORTANT NOTES:
• Start treatment as soon as possible for best results
• Follow the treatment instructions carefully
• Monitor the plant's progress daily
• Remove and destroy severely infected leaves
• Avoid overhead watering to reduce disease spread

PREVENTIVE MEASURES:
• Maintain proper spacing between trees for good air circulation
• Regularly inspect leaves for early signs of disease
• Apply appropriate fungicides preventively during humid seasons
• Keep the orchard clean and free from fallen leaves

⚠️  If the condition worsens after 3-5 days of treatment, please consult with a local agricultural expert or extension officer.

This report was generated by an AI system. For critical agricultural decisions, always verify with human experts.
"""
        
        return report
    
    def generate_technical_report(self, disease_name: str, symptoms: list,
                                treatments: list, confidence_score: float,
                                model_metadata: Dict[str, Any] = None) -> str:
        """Generate a technical report for agricultural professionals"""
        
        prompt = f"""
Act as a plant pathologist and agricultural researcher. Generate a technical report based on the following AI analysis.

TECHNICAL ANALYSIS:
- Disease: {disease_name}
- Model Confidence: {confidence_score:.3f}
- Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
- Treatments: {', '.join(treatments) if treatments else 'Not specified'}

MODEL METADATA:
{json.dumps(model_metadata, indent=2) if model_metadata else 'Not available'}

Provide a detailed technical analysis including:
1. Disease etiology and pathogen information
2. Detailed symptom description
3. Diagnostic methods and differential diagnosis
4. Scientific treatment protocols
5. Epidemiological considerations
6. Research-based prevention strategies
7. Expected treatment outcomes and timelines
8. References to relevant agricultural literature

Use technical terminology appropriate for agricultural professionals and researchers.
"""
        
        if self.gemini_api_key:
            try:
                return self._call_gemini_api(prompt)
            except Exception as e:
                logger.error(f"Technical report generation failed: {str(e)}")
                return "Technical report generation failed. Please try again later."
        else:
            return "LLM service not configured for technical report generation."


# Global LLM service instance
llm_service = LLMService()
