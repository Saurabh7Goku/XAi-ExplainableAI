from typing import Dict, List, Any, Optional
from app.database.database import get_db
from app.database.repositories import DiseaseInfoRepository
from app.utils.exceptions import DatabaseError
from app.utils.logger import logger


class KnowledgeBaseService:
    """Service for managing disease knowledge base"""
    
    def __init__(self):
        self._db = None
        self._repo = None
    
    def _get_repo(self) -> Optional[DiseaseInfoRepository]:
        """Get disease info repository instance"""
        if self._repo is None:
            try:
                self._db = get_db()
                self._repo = DiseaseInfoRepository(self._db)
            except Exception as e:
                logger.error(f"Failed to initialize database repository: {str(e)}")
                return None
        return self._repo
    
    def get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """
        Get disease information by name
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Dictionary containing disease information
        """
        try:
            repo = self._get_repo()
            disease = repo.get_by_name(disease_name)
            
            if disease:
                return {
                    'disease_name': disease.disease_name,
                    'symptoms': disease.symptoms,
                    'treatments': disease.treatments,
                    'description': disease.description,
                    'severity': disease.severity,
                    'prevention_methods': disease.prevention_methods
                }
            else:
                # Return default info if not found in database
                logger.warning(f"Disease '{disease_name}' not found in database, using fallback")
                return self._get_fallback_disease_info(disease_name)
                
        except Exception as e:
            logger.error(f"Failed to get disease info for '{disease_name}': {str(e)}")
            return self._get_fallback_disease_info(disease_name)
    
    def _get_fallback_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Fallback disease information when database is unavailable"""
        
        fallback_info = {
            "Healthy": {
                "symptoms": ["No visible spots or discoloration", "Uniform green color"],
                "treatments": ["Maintain good hygiene", "Regular pruning"],
                "description": "Healthy mango leaf with no signs of disease",
                "severity": "low"
            },
            "Anthracnose": {
                "symptoms": ["Dark sunken lesions on leaves", "Black spores under humid conditions"],
                "treatments": ["Apply copper-based fungicides", "Remove infected parts"],
                "description": "Fungal disease causing dark lesions and leaf spots",
                "severity": "high"
            },
            "Bacterial Canker": {
                "symptoms": ["Water-soaked spots", "Yellow halos around lesions"],
                "treatments": ["Use bactericides like streptomycin", "Avoid overhead watering"],
                "description": "Bacterial infection causing water-soaked lesions",
                "severity": "medium"
            },
            "Cutting Weevil": {
                "symptoms": ["Irregular cuts along leaf margins", "Wilting appearance"],
                "treatments": ["Manual removal of weevils", "Neem oil application"],
                "description": "Insect damage from weevil feeding activity",
                "severity": "low"
            },
            "Die Back": {
                "symptoms": ["Branch tips dying back", "Gradual decline in canopy health"],
                "treatments": ["Prune dead branches", "Improve soil nutrition"],
                "description": "Progressive death of branch tips and twigs",
                "severity": "medium"
            },
            "Gall Midge": {
                "symptoms": ["Small, reddish galls on leaf surface", "Distortion of young leaves"],
                "treatments": ["Prune affected leaves", "Apply appropriate insecticides"],
                "description": "Damage caused by Gall Midge larvae inducing gall formation",
                "severity": "medium"
            },
            "Powdery Mildew": {
                "symptoms": ["White powdery growth on leaves and flowers", "Deformation and premature drop"],
                "treatments": ["Apply sulfur-based fungicides", "Improve air circulation"],
                "description": "Fungal infection creating a white flour-like appearance",
                "severity": "medium"
            },
            "Sooty Mould": {
                "symptoms": ["Black, soot-like fungal growth on surface", "Sticky honeydew residue"],
                "treatments": ["Control sap-sucking insects", "Wash leaves with soap water"],
                "description": "Fungal growth following insect infestations (like aphids or scales)",
                "severity": "low"
            }
        }
        
        return fallback_info.get(disease_name, {
            "symptoms": ["Unknown symptoms"],
            "treatments": ["Consult agricultural expert"],
            "description": f"Disease information for {disease_name} is not available",
            "severity": "unknown"
        })
    
    def get_all_diseases(self) -> List[Dict[str, Any]]:
        """Get all disease information"""
        try:
            repo = self._get_repo()
            diseases = repo.get_all()
            
            return [
                {
                    'disease_name': disease.disease_name,
                    'symptoms': disease.symptoms,
                    'treatments': disease.treatments,
                    'description': disease.description,
                    'severity': disease.severity,
                    'prevention_methods': disease.prevention_methods
                }
                for disease in diseases
            ]
        except Exception as e:
            logger.error(f"Failed to get all diseases: {str(e)}")
            raise DatabaseError(f"Failed to get all diseases: {str(e)}")
    
    def initialize_diseases(self) -> None:
        """Initialize default disease information in database"""
        try:
            repo = self._get_repo()
            if repo:
                repo.initialize_default_diseases()
                logger.info("Default disease information initialized")
            else:
                logger.warning("Skipping disease initialization: Database repository not available")
        except Exception as e:
            logger.warning(f"Failed to initialize diseases in database: {str(e)}. Fallback data will be used.")
    
    def get_treatment_recommendations(self, disease_name: str, severity: str = None) -> List[str]:
        """Get treatment recommendations for a disease"""
        disease_info = self.get_disease_info(disease_name)
        treatments = disease_info.get('treatments', [])
        
        # Add severity-specific recommendations
        if severity == 'high':
            treatments.insert(0, "Immediate action required - consult agricultural expert")
        elif severity == 'medium':
            treatments.append("Monitor progression and re-evaluate in 3-5 days")
        
        return treatments
    
    def get_prevention_methods(self, disease_name: str) -> List[str]:
        """Get prevention methods for a disease"""
        disease_info = self.get_disease_info(disease_name)
        return disease_info.get('prevention_methods', [])
    
    def close(self) -> None:
        """Close database connection"""
        if self._db:
            self._db.close()
            self._db = None
            self._repo = None


# Global knowledge base service instance
knowledge_base_service = KnowledgeBaseService()
