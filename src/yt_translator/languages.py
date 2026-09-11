"""Registry of supported European languages.

Single source of truth: every other module asks *this* module what a language
code means. Adding a language is a one-line change here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """A target language.

    Attributes:
        code: ISO 639-1 code, used by most translation backends.
        name: English display name.
        endonym: The language's name in itself (nice touch for the UI).
        locale: Region-qualified tag. MyMemory insists on these.
    """

    code: str
    name: str
    endonym: str
    locale: str


# Ordered roughly by number of speakers so the UI dropdown is not a lottery.
_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "English", "en-GB"),
    Language("ru", "Russian", "Русский", "ru-RU"),
    Language("de", "German", "Deutsch", "de-DE"),
    Language("fr", "French", "Français", "fr-FR"),
    Language("it", "Italian", "Italiano", "it-IT"),
    Language("es", "Spanish", "Español", "es-ES"),
    Language("pl", "Polish", "Polski", "pl-PL"),
    Language("uk", "Ukrainian", "Українська", "uk-UA"),
    Language("ro", "Romanian", "Română", "ro-RO"),
    Language("nl", "Dutch", "Nederlands", "nl-NL"),
    Language("pt", "Portuguese", "Português", "pt-PT"),
    Language("el", "Greek", "Ελληνικά", "el-GR"),
    Language("hu", "Hungarian", "Magyar", "hu-HU"),
    Language("cs", "Czech", "Čeština", "cs-CZ"),
    Language("sv", "Swedish", "Svenska", "sv-SE"),
    Language("bg", "Bulgarian", "Български", "bg-BG"),
    Language("da", "Danish", "Dansk", "da-DK"),
    Language("fi", "Finnish", "Suomi", "fi-FI"),
    Language("sk", "Slovak", "Slovenčina", "sk-SK"),
    Language("no", "Norwegian", "Norsk", "no-NO"),
    Language("hr", "Croatian", "Hrvatski", "hr-HR"),
    Language("sr", "Serbian", "Српски", "sr-RS"),
    Language("lt", "Lithuanian", "Lietuvių", "lt-LT"),
    Language("sl", "Slovenian", "Slovenščina", "sl-SI"),
    Language("lv", "Latvian", "Latviešu", "lv-LV"),
    Language("et", "Estonian", "Eesti", "et-EE"),
    Language("mk", "Macedonian", "Македонски", "mk-MK"),
    Language("sq", "Albanian", "Shqip", "sq-AL"),
    Language("bs", "Bosnian", "Bosanski", "bs-BA"),
    Language("ca", "Catalan", "Català", "ca-ES"),
    Language("ga", "Irish", "Gaeilge", "ga-IE"),
    Language("cy", "Welsh", "Cymraeg", "cy-GB"),
    Language("mt", "Maltese", "Malti", "mt-MT"),
    Language("is", "Icelandic", "Íslenska", "is-IS"),
    Language("eu", "Basque", "Euskara", "eu-ES"),
    Language("gl", "Galician", "Galego", "gl-ES"),
    Language("lb", "Luxembourgish", "Lëtzebuergesch", "lb-LU"),
    Language("be", "Belarusian", "Беларуская", "be-BY"),
)

_BY_CODE: dict[str, Language] = {lang.code: lang for lang in _LANGUAGES}
_BY_NAME: dict[str, Language] = {lang.name.casefold(): lang for lang in _LANGUAGES}


class UnknownLanguageError(ValueError):
    """Raised when a requested language is not in the registry."""


def all_languages() -> tuple[Language, ...]:
    """Every supported language, in display order."""
    return _LANGUAGES


def resolve(value: str) -> Language:
    """Look up a language by ISO code or English name, case-insensitively.

    Args:
        value: e.g. ``"de"``, ``"DE"``, ``"German"``, ``"german"``.

    Returns:
        The matching :class:`Language`.

    Raises:
        UnknownLanguageError: If nothing matches.
    """
    key = value.strip().casefold()
    found = _BY_CODE.get(key) or _BY_NAME.get(key)
    if found is None:
        raise UnknownLanguageError(
            f"{value!r} is not a supported European language. "
            f"Try one of: {', '.join(sorted(_BY_CODE))}"
        )
    return found


def is_supported(value: str) -> bool:
    """True if :func:`resolve` would succeed for ``value``."""
    try:
        resolve(value)
    except UnknownLanguageError:
        return False
    return True
