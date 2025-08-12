"""
Internationalization (i18n) module for photonic AI systems.

Provides multi-language support, locale-specific formatting,
and cultural adaptation for global photonic neural network deployments.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import locale
from pathlib import Path

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


class Region(Enum):
    """Supported regions with specific requirements."""
    NORTH_AMERICA = "NA"
    EUROPE = "EU"
    ASIA_PACIFIC = "APAC"
    LATIN_AMERICA = "LATAM"
    MIDDLE_EAST = "ME"
    AFRICA = "AF"


@dataclass
class LocaleConfig:
    """Locale-specific configuration."""
    language: SupportedLanguage
    region: Region
    currency: str
    date_format: str
    number_format: str
    timezone: str
    rtl: bool = False  # Right-to-left text direction
    
    @property
    def locale_code(self) -> str:
        """Get standard locale code."""
        region_codes = {
            Region.NORTH_AMERICA: "US",
            Region.EUROPE: "EU",
            Region.ASIA_PACIFIC: "JP",
            Region.LATIN_AMERICA: "MX",
            Region.MIDDLE_EAST: "AE",
            Region.AFRICA: "ZA"
        }
        return f"{self.language.value}_{region_codes.get(self.region, 'US')}"


class TranslationManager:
    """
    Translation management for multi-language support.
    
    Manages loading, caching, and retrieval of translated strings
    with fallback support and interpolation.
    """
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        """Initialize translation manager."""
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self.fallback_chain = [SupportedLanguage.ENGLISH]
        
        # Load default translations
        self._load_built_in_translations()
    
    def _load_built_in_translations(self):
        """Load built-in translations for photonic AI terminology."""
        # English (base language)
        self.translations[SupportedLanguage.ENGLISH] = {
            # System Messages
            "system.startup": "Photonic AI Simulator starting up...",
            "system.ready": "System ready for operations",
            "system.shutdown": "Shutting down photonic AI system",
            "system.error": "System error occurred",
            
            # Model Operations
            "model.loading": "Loading photonic neural network model",
            "model.loaded": "Model loaded successfully",
            "model.training": "Training photonic neural network",
            "model.inference": "Performing inference",
            "model.validation": "Validating model performance",
            
            # Hardware Status
            "hardware.temperature": "Temperature",
            "hardware.power": "Power consumption",
            "hardware.latency": "Inference latency",
            "hardware.wavelength": "Wavelength",
            "hardware.thermal_drift": "Thermal drift",
            
            # Performance Metrics
            "metrics.accuracy": "Accuracy",
            "metrics.throughput": "Throughput",
            "metrics.efficiency": "Energy efficiency",
            "metrics.speedup": "Speedup factor",
            
            # Error Messages
            "error.invalid_input": "Invalid input data provided",
            "error.model_not_found": "Model not found",
            "error.training_failed": "Training process failed",
            "error.hardware_failure": "Hardware component failure",
            "error.thermal_limit": "Thermal limit exceeded",
            
            # Units
            "units.nanoseconds": "ns",
            "units.milliwatts": "mW",
            "units.kelvin": "K",
            "units.nanometers": "nm",
            "units.picometers": "pm",
            "units.samples_per_second": "samples/sec",
            
            # User Interface
            "ui.start": "Start",
            "ui.stop": "Stop",
            "ui.reset": "Reset",
            "ui.configure": "Configure",
            "ui.monitor": "Monitor",
            "ui.export": "Export",
            "ui.import": "Import",
            
            # Status
            "status.running": "Running",
            "status.stopped": "Stopped",
            "status.error": "Error",
            "status.warning": "Warning",
            "status.healthy": "Healthy",
            "status.degraded": "Degraded"
        }
        
        # Spanish translations
        self.translations[SupportedLanguage.SPANISH] = {
            "system.startup": "Iniciando Simulador de IA Fotónica...",
            "system.ready": "Sistema listo para operaciones",
            "system.shutdown": "Apagando sistema de IA fotónica",
            "system.error": "Error del sistema ocurrido",
            
            "model.loading": "Cargando modelo de red neuronal fotónica",
            "model.loaded": "Modelo cargado exitosamente",
            "model.training": "Entrenando red neuronal fotónica",
            "model.inference": "Realizando inferencia",
            "model.validation": "Validando rendimiento del modelo",
            
            "hardware.temperature": "Temperatura",
            "hardware.power": "Consumo de energía",
            "hardware.latency": "Latencia de inferencia",
            "hardware.wavelength": "Longitud de onda",
            "hardware.thermal_drift": "Deriva térmica",
            
            "metrics.accuracy": "Precisión",
            "metrics.throughput": "Rendimiento",
            "metrics.efficiency": "Eficiencia energética",
            "metrics.speedup": "Factor de aceleración",
            
            "error.invalid_input": "Datos de entrada inválidos proporcionados",
            "error.model_not_found": "Modelo no encontrado",
            "error.training_failed": "Proceso de entrenamiento falló",
            "error.hardware_failure": "Falla del componente de hardware",
            "error.thermal_limit": "Límite térmico excedido",
            
            "status.running": "Ejecutándose",
            "status.stopped": "Detenido",
            "status.error": "Error",
            "status.healthy": "Saludable"
        }
        
        # French translations
        self.translations[SupportedLanguage.FRENCH] = {
            "system.startup": "Démarrage du Simulateur d'IA Photonique...",
            "system.ready": "Système prêt pour les opérations",
            "system.shutdown": "Arrêt du système d'IA photonique",
            "system.error": "Erreur système survenue",
            
            "model.loading": "Chargement du modèle de réseau neuronal photonique",
            "model.loaded": "Modèle chargé avec succès",
            "model.training": "Entraînement du réseau neuronal photonique",
            "model.inference": "Exécution de l'inférence",
            "model.validation": "Validation des performances du modèle",
            
            "hardware.temperature": "Température",
            "hardware.power": "Consommation d'énergie",
            "hardware.latency": "Latence d'inférence",
            "hardware.wavelength": "Longueur d'onde",
            "hardware.thermal_drift": "Dérive thermique",
            
            "metrics.accuracy": "Précision",
            "metrics.throughput": "Débit",
            "metrics.efficiency": "Efficacité énergétique",
            "metrics.speedup": "Facteur d'accélération",
            
            "error.invalid_input": "Données d'entrée invalides fournies",
            "error.model_not_found": "Modèle non trouvé",
            "error.training_failed": "Processus d'entraînement échoué",
            "error.hardware_failure": "Défaillance du composant matériel",
            "error.thermal_limit": "Limite thermique dépassée",
            
            "status.running": "En cours",
            "status.stopped": "Arrêté",
            "status.error": "Erreur",
            "status.healthy": "Sain"
        }
        
        # German translations
        self.translations[SupportedLanguage.GERMAN] = {
            "system.startup": "Photonischer KI-Simulator startet...",
            "system.ready": "System bereit für Operationen",
            "system.shutdown": "Photonisches KI-System wird heruntergefahren",
            "system.error": "Systemfehler aufgetreten",
            
            "model.loading": "Lade photonisches neuronales Netzwerk-Modell",
            "model.loaded": "Modell erfolgreich geladen",
            "model.training": "Trainiere photonisches neuronales Netzwerk",
            "model.inference": "Führe Inferenz durch",
            "model.validation": "Validiere Modellleistung",
            
            "hardware.temperature": "Temperatur",
            "hardware.power": "Energieverbrauch",
            "hardware.latency": "Inferenz-Latenz",
            "hardware.wavelength": "Wellenlänge",
            "hardware.thermal_drift": "Thermische Drift",
            
            "metrics.accuracy": "Genauigkeit",
            "metrics.throughput": "Durchsatz",
            "metrics.efficiency": "Energieeffizienz",
            "metrics.speedup": "Beschleunigungsfaktor",
            
            "error.invalid_input": "Ungültige Eingabedaten bereitgestellt",
            "error.model_not_found": "Modell nicht gefunden",
            "error.training_failed": "Trainingsprozess fehlgeschlagen",
            "error.hardware_failure": "Hardware-Komponentenfehler",
            "error.thermal_limit": "Thermische Grenze überschritten",
            
            "status.running": "Läuft",
            "status.stopped": "Gestoppt",
            "status.error": "Fehler",
            "status.healthy": "Gesund"
        }
        
        # Japanese translations
        self.translations[SupportedLanguage.JAPANESE] = {
            "system.startup": "フォトニックAIシミュレータを起動中...",
            "system.ready": "システムは動作準備完了",
            "system.shutdown": "フォトニックAIシステムをシャットダウン中",
            "system.error": "システムエラーが発生しました",
            
            "model.loading": "フォトニック神経ネットワークモデルを読み込み中",
            "model.loaded": "モデルが正常に読み込まれました",
            "model.training": "フォトニック神経ネットワークを訓練中",
            "model.inference": "推論を実行中",
            "model.validation": "モデル性能を検証中",
            
            "hardware.temperature": "温度",
            "hardware.power": "消費電力",
            "hardware.latency": "推論遅延",
            "hardware.wavelength": "波長",
            "hardware.thermal_drift": "熱ドリフト",
            
            "metrics.accuracy": "精度",
            "metrics.throughput": "スループット",
            "metrics.efficiency": "エネルギー効率",
            "metrics.speedup": "高速化倍率",
            
            "error.invalid_input": "無効な入力データが提供されました",
            "error.model_not_found": "モデルが見つかりません",
            "error.training_failed": "訓練プロセスが失敗しました",
            "error.hardware_failure": "ハードウェアコンポーネントの故障",
            "error.thermal_limit": "熱限界を超えました",
            
            "status.running": "実行中",
            "status.stopped": "停止",
            "status.error": "エラー",
            "status.healthy": "正常"
        }
        
        # Chinese Simplified translations
        self.translations[SupportedLanguage.CHINESE_SIMPLIFIED] = {
            "system.startup": "光子AI仿真器启动中...",
            "system.ready": "系统准备就绪",
            "system.shutdown": "关闭光子AI系统",
            "system.error": "系统错误发生",
            
            "model.loading": "加载光子神经网络模型",
            "model.loaded": "模型加载成功",
            "model.training": "训练光子神经网络",
            "model.inference": "执行推理",
            "model.validation": "验证模型性能",
            
            "hardware.temperature": "温度",
            "hardware.power": "功耗",
            "hardware.latency": "推理延迟",
            "hardware.wavelength": "波长",
            "hardware.thermal_drift": "热漂移",
            
            "metrics.accuracy": "准确率",
            "metrics.throughput": "吞吐量",
            "metrics.efficiency": "能效",
            "metrics.speedup": "加速倍数",
            
            "error.invalid_input": "提供了无效的输入数据",
            "error.model_not_found": "未找到模型",
            "error.training_failed": "训练过程失败",
            "error.hardware_failure": "硬件组件故障",
            "error.thermal_limit": "超过热限制",
            
            "status.running": "运行中",
            "status.stopped": "已停止",
            "status.error": "错误",
            "status.healthy": "健康"
        }
    
    def set_language(self, language: SupportedLanguage):
        """Set current language."""
        self.current_language = language
        logger.info(f"Language set to {language.value}")
    
    def get_text(self, key: str, language: SupportedLanguage = None, 
                **kwargs) -> str:
        """
        Get translated text for given key.
        
        Args:
            key: Translation key
            language: Language to use (None for current)
            **kwargs: Values for string interpolation
            
        Returns:
            Translated and interpolated text
        """
        language = language or self.current_language
        
        # Try to get translation
        if language in self.translations:
            if key in self.translations[language]:
                text = self.translations[language][key]
                return self._interpolate(text, kwargs)
        
        # Try fallback languages
        for fallback_lang in self.fallback_chain:
            if fallback_lang in self.translations:
                if key in self.translations[fallback_lang]:
                    text = self.translations[fallback_lang][key]
                    return self._interpolate(text, kwargs)
        
        # Return key as fallback
        logger.warning(f"Translation not found for key: {key}")
        return key
    
    def _interpolate(self, text: str, values: Dict[str, Any]) -> str:
        """Interpolate values into text template."""
        if not values:
            return text
        
        try:
            return text.format(**values)
        except (KeyError, ValueError) as e:
            logger.warning(f"String interpolation failed: {e}")
            return text
    
    def load_custom_translations(self, language: SupportedLanguage, 
                               translations: Dict[str, str]):
        """Load custom translations for a language."""
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language].update(translations)
        logger.info(f"Loaded {len(translations)} custom translations for {language.value}")
    
    def export_translations(self, language: SupportedLanguage) -> Dict[str, str]:
        """Export translations for a language."""
        return self.translations.get(language, {}).copy()


class LocaleFormatter:
    """
    Locale-specific formatting for numbers, dates, and currencies.
    """
    
    def __init__(self, locale_config: LocaleConfig):
        """Initialize formatter with locale configuration."""
        self.config = locale_config
        self._setup_locale()
    
    def _setup_locale(self):
        """Set up system locale if available."""
        try:
            locale.setlocale(locale.LC_ALL, self.config.locale_code)
        except locale.Error:
            logger.warning(f"Locale {self.config.locale_code} not available, using default")
    
    def format_number(self, number: Union[int, float], 
                     decimal_places: int = 2) -> str:
        """Format number according to locale."""
        if self.config.language == SupportedLanguage.GERMAN:
            # German uses comma for decimal separator, period for thousands
            formatted = f"{number:,.{decimal_places}f}"
            return formatted.replace(',', 'X').replace('.', ',').replace('X', '.')
        
        elif self.config.language in [SupportedLanguage.FRENCH, SupportedLanguage.SPANISH]:
            # French and Spanish use space for thousands separator
            formatted = f"{number:,.{decimal_places}f}"
            return formatted.replace(',', ' ')
        
        else:
            # Default (English) formatting
            return f"{number:,.{decimal_places}f}"
    
    def format_percentage(self, value: float, decimal_places: int = 1) -> str:
        """Format percentage according to locale."""
        percentage = value * 100
        formatted_number = self.format_number(percentage, decimal_places)
        
        if self.config.language == SupportedLanguage.FRENCH:
            return f"{formatted_number} %"  # Space before %
        else:
            return f"{formatted_number}%"
    
    def format_currency(self, amount: float, currency_code: str = None) -> str:
        """Format currency according to locale."""
        currency = currency_code or self.config.currency
        formatted_amount = self.format_number(amount, 2)
        
        if self.config.language == SupportedLanguage.JAPANESE:
            return f"¥{formatted_amount}"
        elif self.config.language == SupportedLanguage.CHINESE_SIMPLIFIED:
            return f"¥{formatted_amount}"
        elif currency == "EUR":
            if self.config.language == SupportedLanguage.GERMAN:
                return f"{formatted_amount} €"
            else:
                return f"€{formatted_amount}"
        elif currency == "USD":
            return f"${formatted_amount}"
        else:
            return f"{formatted_amount} {currency}"
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to locale."""
        if self.config.language == SupportedLanguage.JAPANESE:
            return dt.strftime("%Y年%m月%d日 %H:%M:%S")
        elif self.config.language == SupportedLanguage.CHINESE_SIMPLIFIED:
            return dt.strftime("%Y年%m月%d日 %H:%M:%S")
        elif self.config.language == SupportedLanguage.GERMAN:
            return dt.strftime("%d.%m.%Y %H:%M:%S")
        elif self.config.language == SupportedLanguage.FRENCH:
            return dt.strftime("%d/%m/%Y %H:%M:%S")
        else:
            # Default (English) format
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def format_scientific_notation(self, value: float, precision: int = 2) -> str:
        """Format scientific notation according to locale."""
        formatted = f"{value:.{precision}e}"
        
        if self.config.language == SupportedLanguage.GERMAN:
            # German uses comma for decimal separator
            return formatted.replace('.', ',')
        else:
            return formatted


class PhotonicI18nSystem:
    """
    Comprehensive internationalization system for photonic AI.
    
    Integrates translation management, locale formatting, and
    cultural adaptation for global deployments.
    """
    
    def __init__(self, default_locale: LocaleConfig = None):
        """Initialize i18n system."""
        self.default_locale = default_locale or LocaleConfig(
            language=SupportedLanguage.ENGLISH,
            region=Region.NORTH_AMERICA,
            currency="USD",
            date_format="%Y-%m-%d",
            number_format="1,234.56",
            timezone="UTC"
        )
        
        self.current_locale = self.default_locale
        self.translator = TranslationManager(self.default_locale.language)
        self.formatter = LocaleFormatter(self.default_locale)
        
    def set_locale(self, locale_config: LocaleConfig):
        """Set current locale configuration."""
        self.current_locale = locale_config
        self.translator.set_language(locale_config.language)
        self.formatter = LocaleFormatter(locale_config)
        
        logger.info(f"Locale set to {locale_config.language.value}_{locale_config.region.value}")
    
    def t(self, key: str, **kwargs) -> str:
        """
        Translate text (shorthand method).
        
        Args:
            key: Translation key
            **kwargs: Values for interpolation
            
        Returns:
            Translated text
        """
        return self.translator.get_text(key, **kwargs)
    
    def format_metric(self, metric_name: str, value: Union[int, float], 
                     unit: str = None) -> str:
        """
        Format performance metric with proper localization.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            unit: Unit of measurement
            
        Returns:
            Formatted metric string
        """
        # Translate metric name
        translated_name = self.t(f"metrics.{metric_name}")
        
        # Format value
        if isinstance(value, float):
            if abs(value) >= 1e6 or abs(value) <= 1e-6:
                formatted_value = self.formatter.format_scientific_notation(value)
            else:
                formatted_value = self.formatter.format_number(value)
        else:
            formatted_value = self.formatter.format_number(float(value), 0)
        
        # Translate unit if provided
        unit_text = ""
        if unit:
            unit_key = f"units.{unit.lower().replace('/', '_per_').replace(' ', '_')}"
            translated_unit = self.t(unit_key)
            if translated_unit != unit_key:  # Translation found
                unit_text = f" {translated_unit}"
            else:
                unit_text = f" {unit}"
        
        return f"{translated_name}: {formatted_value}{unit_text}"
    
    def format_status_message(self, component: str, status: str, 
                            additional_info: Dict[str, Any] = None) -> str:
        """
        Format status message with localization.
        
        Args:
            component: Component name
            status: Status value
            additional_info: Additional information
            
        Returns:
            Formatted status message
        """
        # Translate status
        status_key = f"status.{status.lower()}"
        translated_status = self.t(status_key)
        
        # Format base message
        message = f"{component}: {translated_status}"
        
        # Add additional information if provided
        if additional_info:
            details = []
            for key, value in additional_info.items():
                if isinstance(value, (int, float)):
                    formatted_value = self.formatter.format_number(value)
                    details.append(f"{key}={formatted_value}")
                else:
                    details.append(f"{key}={value}")
            
            if details:
                message += f" ({', '.join(details)})"
        
        return message
    
    def get_error_message(self, error_type: str, **kwargs) -> str:
        """
        Get localized error message.
        
        Args:
            error_type: Type of error
            **kwargs: Error details for interpolation
            
        Returns:
            Localized error message
        """
        error_key = f"error.{error_type}"
        return self.t(error_key, **kwargs)
    
    def export_locale_data(self) -> Dict[str, Any]:
        """Export current locale configuration and translations."""
        return {
            "locale_config": {
                "language": self.current_locale.language.value,
                "region": self.current_locale.region.value,
                "currency": self.current_locale.currency,
                "date_format": self.current_locale.date_format,
                "number_format": self.current_locale.number_format,
                "timezone": self.current_locale.timezone,
                "rtl": self.current_locale.rtl
            },
            "translations": self.translator.export_translations(self.current_locale.language),
            "supported_languages": [lang.value for lang in SupportedLanguage],
            "supported_regions": [region.value for region in Region]
        }


# Global i18n instance
_global_i18n = None


def get_i18n_system() -> PhotonicI18nSystem:
    """Get global i18n system instance."""
    global _global_i18n
    if _global_i18n is None:
        _global_i18n = PhotonicI18nSystem()
    return _global_i18n


def init_i18n(locale_config: LocaleConfig = None):
    """Initialize global i18n system."""
    global _global_i18n
    _global_i18n = PhotonicI18nSystem(locale_config)
    return _global_i18n


def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return get_i18n_system().t(key, **kwargs)


def create_locale_config(language_code: str, region_code: str = None) -> LocaleConfig:
    """
    Create locale configuration from language and region codes.
    
    Args:
        language_code: ISO language code (e.g., 'en', 'ja', 'zh-CN')
        region_code: Region code (e.g., 'US', 'EU', 'APAC')
        
    Returns:
        Locale configuration
    """
    # Map language codes to enums
    language_map = {
        "en": SupportedLanguage.ENGLISH,
        "es": SupportedLanguage.SPANISH,
        "fr": SupportedLanguage.FRENCH,
        "de": SupportedLanguage.GERMAN,
        "ja": SupportedLanguage.JAPANESE,
        "zh-CN": SupportedLanguage.CHINESE_SIMPLIFIED,
        "zh-TW": SupportedLanguage.CHINESE_TRADITIONAL,
        "ko": SupportedLanguage.KOREAN,
        "pt": SupportedLanguage.PORTUGUESE,
        "it": SupportedLanguage.ITALIAN,
        "ru": SupportedLanguage.RUSSIAN,
        "ar": SupportedLanguage.ARABIC,
        "hi": SupportedLanguage.HINDI
    }
    
    # Map region codes to enums
    region_map = {
        "NA": Region.NORTH_AMERICA,
        "US": Region.NORTH_AMERICA,
        "EU": Region.EUROPE,
        "APAC": Region.ASIA_PACIFIC,
        "LATAM": Region.LATIN_AMERICA,
        "ME": Region.MIDDLE_EAST,
        "AF": Region.AFRICA
    }
    
    language = language_map.get(language_code, SupportedLanguage.ENGLISH)
    region = region_map.get(region_code, Region.NORTH_AMERICA)
    
    # Set defaults based on language/region
    currency = "USD"
    timezone = "UTC"
    date_format = "%Y-%m-%d"
    number_format = "1,234.56"
    rtl = False
    
    if language == SupportedLanguage.JAPANESE:
        currency = "JPY"
        timezone = "Asia/Tokyo"
        date_format = "%Y年%m月%d日"
    elif language in [SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.CHINESE_TRADITIONAL]:
        currency = "CNY"
        timezone = "Asia/Shanghai"
        date_format = "%Y年%m月%d日"
    elif region == Region.EUROPE:
        currency = "EUR"
        timezone = "Europe/Berlin"
        date_format = "%d.%m.%Y"
        number_format = "1.234,56"
    elif language == SupportedLanguage.ARABIC:
        currency = "AED"
        timezone = "Asia/Dubai"
        rtl = True
    
    return LocaleConfig(
        language=language,
        region=region,
        currency=currency,
        date_format=date_format,
        number_format=number_format,
        timezone=timezone,
        rtl=rtl
    )