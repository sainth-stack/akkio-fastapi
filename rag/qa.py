import os
import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, unquote

from openai import OpenAI

from .chroma_store import query_chunks


LOGGER_NAME = "uae_legislation_ingest"
logger = logging.getLogger(LOGGER_NAME)


def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic or English."""
    if not text:
        return "en"
    # Count Arabic characters (Unicode range 0600-06FF)
    ar_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    # Count English/Latin characters
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # If more Arabic than English (with threshold), consider it Arabic
    if ar_chars > en_chars * 0.5:
        return "ar"
    return "en"


def normalize_ar(text: str) -> str:
    if not text:
        return text
    # Remove tatweel and diacritics; normalize alef/ya variations
    text = re.sub("[\u064B-\u065F\u0670]", "", text)  # harakat
    text = text.replace("\u0640", "")  # tatweel
    text = re.sub("[\u0622\u0623\u0625]", "\u0627", text)  # ALEF variants -> ALEF
    text = text.replace("\u0649", "\u064A")  # ALEF MAKSURA -> YA
    return text


SYSTEM_PROMPT_AR = (
    "أنت مساعد قانوني متخصص في القانون الإماراتي. استخدم المقاطع الموفّرة لتقديم إجابة شاملة ومفصلة. "
    "إذا كان هناك محتوى ذو صلة في المقاطع، قدم شرحاً مفصلاً بناءً عليه. "
    "اذكر رقم القانون والمادة إن وُجدا، وقدم معلومات واضحة ومنظمة. "
    "استخدم المقاطع لتفسير المفاهيم القانونية وإعطاء أمثلة. فقط إذا كان المحتوى الموفّر لا يحتوي على أي معلومات ذات صلة بالسؤال، أذكر ذلك بوضوح. "
    "\n\n⚠️ مهم جداً: يجب أن تكون إجابتك بصيغة HTML باستخدام العلامات التالية:\n"
    "- استخدم <h3> للعناوين الرئيسية\n"
    "- استخدم <h4> للعناوين الفرعية\n"
    "- استخدم <p> للفقرات\n"
    "- استخدم <ul> و <li> للقوائم\n"
    "- استخدم <strong> للتأكيد\n"
    "- استخدم <em> للنص المائل\n"
    "- استخدم <hr> لفصل الأقسام\n"
    "لا تستخدم markdown. استخدم HTML فقط."
)

SYSTEM_PROMPT_EN = (
    "You are a specialized legal assistant for UAE law. Use the provided excerpts to deliver a comprehensive and detailed answer. "
    "If there is relevant content in the excerpts, provide a detailed explanation based on it. "
    "Mention law numbers and articles when available, and provide clear, well-organized information. "
    "Use the excerpts to explain legal concepts and provide examples. Only if the provided content contains no relevant information to the question, state that clearly. "
    "\n\n⚠️ CRITICALLY IMPORTANT: Your response MUST be formatted in HTML using the following tags:\n"
    "- Use <h3> for main titles\n"
    "- Use <h4> for subtitles/section headings\n"
    "- Use <p> for paragraphs\n"
    "- Use <ul> and <li> for lists\n"
    "- Use <strong> for emphasis/bold text\n"
    "- Use <em> for italics\n"
    "- Use <hr> for section separators\n"
    "- Structure your response with clear sections and subsections\n"
    "Do NOT use markdown formatting (no **, ##, -, etc.). Use ONLY HTML tags."
)


ACTION_SYSTEM_PROMPT_AR = (
    "أنت مساعد قانوني متخصص في استخراج الإجراءات العملية والعقوبات والمتطلبات الزمنية من النصوص القانونية الإماراتية. "
    "استخدم المقاطع الموفّرة لإنتاج إجابة مركزة على (الإجراء/العقوبة/المدة) المطلوبة في السؤال. "
    "إذا تضمنت النصوص مدد زمنية (سنوات/أشهر/أيام) أو غرامات أو خطوات إجرائية، فاذكرها بدقة مع رقم القانون والمادة إن أمكن. "
    "\n\n⚠️ مهم: يجب أن تكون الإجابة بصيغة HTML فقط باستخدام العلامات التالية: "
    "<h3> و <h4> و <p> و <ul> و <li> و <strong> و <em> و <hr>. "
    "ابدأ بخلاصة مباشرة ومحددة (مثلاً: <h3>النتيجة: سنتان</h3>) ثم قدم التفاصيل في قائمة نقطية مرتبة. لا تستخدم Markdown."
)

ACTION_SYSTEM_PROMPT_EN = (
    "You are a legal assistant focused on extracting concrete actions, penalties, and time limits from UAE legal texts. "
    "Use the provided excerpts to produce a concise, action-oriented answer to the user's request. "
    "If the law specifies durations (years/months/days), fines, or procedural steps, state them precisely and reference law/article numbers when available. "
    "\n\n⚠️ IMPORTANT: Output must be HTML only using <h3>, <h4>, <p>, <ul>, <li>, <strong>, <em>, <hr>. "
    "Start with a direct, short result (e.g., <h3>Result: Two years</h3>) followed by a bullet list of details. Do NOT use Markdown."
)


def is_action_query(query: str, lang: str) -> bool:
    """Lightweight heuristic to detect action/penalty/duration oriented queries.

    This avoids an extra model call by matching indicative keywords in EN/AR.
    """
    if not query:
        return False
    q = query.lower()
    if lang == "ar":
        patterns = [
            "الإجراءات", "إجراء", "العقوبة", "غرامة", "مدة", "سنوات", "أشهر", "أيام",
            "كم سنة", "كم شهر", "ما هي العقوبة", "ما العقوبة", "ما الغرامة", "ما الإجراء",
            "عقوبة", "جزاء", "فترة", "مهلة", "الحد", "الجزاءات",
        ]
    else:
        patterns = [
            "action", "actions", "penalty", "penalties", "fine", "punishment", "sentence",
            "how many years", "how many months", "time limit", "deadline", "period", "duration",
            "what is the penalty", "what is the fine", "what is the sentence", "shall", "must",
        ]
    return any(p in q for p in patterns)


def _extract_title_from_url(url: str) -> str:
    """Extract a meaningful title from URL if title is missing or just 'ع'."""
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        # Extract legislation ID or filename from path
        if "/legislations/" in path:
            parts = path.split("/legislations/")
            if len(parts) > 1:
                leg_id = parts[1].split("/")[0]
                return f"UAE Legislation {leg_id}"
        if "/download" in path:
            parts = path.split("/download")[0].split("/")
            if parts:
                return parts[-1].replace("-", " ").title() or "Legal Document"
        filename = os.path.basename(path)
        if filename and filename != "download":
            return unquote(filename).replace(".pdf", "").replace("-", " ").title()
        return "UAE Legal Document"
    except Exception:
        return "UAE Legal Document"


def build_prompt(chunks: List[Dict[str, Any]], user_query: str, lang: str = "en") -> str:
    if lang == "ar":
        lines = ["السياق:"]
        for i, ch in enumerate(chunks, 1):
            meta = ch.get("metadata", {})
            title = meta.get("title") or _extract_title_from_url(meta.get("url", ""))
            # If title is just a single Arabic character like "ع", use URL-based title
            if title and len(title.strip()) == 1 and ord(title.strip()[0]) >= 0x0600:
                title = _extract_title_from_url(meta.get("url", ""))
            law_no = meta.get("law_number") or ""
            article = meta.get("article") or ""
            url = meta.get("url") or ""
            snippet = ch.get("document", "").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            lines.append(f"[مقتطف {i}] {title} (قانون {law_no}) {article} | {url}\n{snippet}")
        lines.append("\nالسؤال:")
        lines.append(user_query)
        lines.append("\nتعليمات: أجب بالعربية بتفصيل ووضوح. استخدم جميع المعلومات المتاحة في المقاطع لشرح السؤال بشكل شامل. قدم أمثلة واستشهادات مباشرة من النصوص. إذا طُلب منك الشرح أو التوضيح، قدم إجابة مفصلة ومنظمة. قم بتنسيق إجابتك باستخدام علامات HTML (<h3>, <h4>, <p>, <ul>, <li>, <strong>, <em>, <hr>). لا تستخدم markdown - استخدم HTML فقط.")
    else:
        lines = ["Context:"]
        for i, ch in enumerate(chunks, 1):
            meta = ch.get("metadata", {})
            title = meta.get("title") or _extract_title_from_url(meta.get("url", ""))
            # If title is just a single Arabic character like "ع", use URL-based title
            if title and len(title.strip()) == 1 and ord(title.strip()[0]) >= 0x0600:
                title = _extract_title_from_url(meta.get("url", ""))
            law_no = meta.get("law_number") or ""
            article = meta.get("article") or ""
            url = meta.get("url") or ""
            snippet = ch.get("document", "").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            lines.append(f"[Excerpt {i}] {title} (Law {law_no}) {article} | {url}\n{snippet}")
        lines.append("\nQuestion:")
        lines.append(user_query)
        lines.append("\nInstructions: Answer in English with detail and clarity. Use all available information from the excerpts to comprehensively explain the question. Provide examples and direct citations from the texts. If asked to explain or clarify, provide a detailed, well-structured answer. Format your response using HTML tags (<h3>, <h4>, <p>, <ul>, <li>, <strong>, <em>, <hr>). Do NOT use markdown - use HTML only.")
    return "\n".join(lines)


def answer_with_rag(query: str, top_k: int = 8, where: Optional[Dict[str, Any]] = None, mode: Optional[str] = None) -> Dict[str, Any]:
    lang = detect_language(query)
    # Normalize query based on language
    qn = normalize_ar(query) if lang == "ar" else query
    
    res = query_chunks(qn, top_k=top_k, where=where)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Pair and sort by distance (ascending)
    pairs = [
        {"document": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]
    pairs.sort(key=lambda x: x.get("distance", 1.0))

    # Determine intent: action vs answer
    effective_mode = (mode or "auto").lower()
    if effective_mode == "auto":
        effective_mode = "action" if is_action_query(query, lang) else "answer"

    # Build prompts
    prompt = build_prompt(pairs, query, lang=lang)
    if effective_mode == "action":
        system_prompt = ACTION_SYSTEM_PROMPT_AR if lang == "ar" else ACTION_SYSTEM_PROMPT_EN
    else:
        system_prompt = SYSTEM_PROMPT_AR if lang == "ar" else SYSTEM_PROMPT_EN

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Use slightly higher temperature for explanation queries to get more detailed responses
    # Detect if query asks for explanation
    is_explanation_query = any(word in query.lower() for word in ["explain", "what is", "describe", "tell me about", "clarify", "شرح", "اشرح", "ما هو", "وضح"])
    temperature = 0.4 if is_explanation_query else 0.3
    
    completion = client.chat.completions.create(
        model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=2000,  # Allow longer responses for detailed explanations
    )
    answer = completion.choices[0].message.content

    # Deduplicate citations based on unique combination of pdf_url, title, law_number, article, and pages
    citations_dict = {}
    for p in pairs:
        m = p.get("metadata", {})
        title = m.get("title")
        # Fix title if it's just a single Arabic character
        if title and len(title.strip()) == 1 and ord(title.strip()[0]) >= 0x0600:
            title = _extract_title_from_url(m.get("url", ""))
        title = title or _extract_title_from_url(m.get("url", ""))
        pdf_url = m.get("url")
        law_number = m.get("law_number")
        article = m.get("article")
        pages = m.get("pages")
        
        # Create unique key for deduplication (using pdf_url as primary key since it's most unique)
        citation_key = pdf_url or ""
        
        # Only add if not already present and has a valid pdf_url
        if citation_key not in citations_dict and pdf_url:
            citations_dict[citation_key] = {
                "title": title,
                "law_number": law_number,
                "article": article,
                "pages": pages,
                "pdf_url": pdf_url,
            }
    
    citations = list(citations_dict.values())

    return {
        "answer": answer,
        "language": lang,
        "mode": effective_mode,
        "citations": citations,
        "chunks_preview": [{"text": p.get("document"), "metadata": p.get("metadata")} for p in pairs],
    }


__all__ = ["normalize_ar", "answer_with_rag", "build_prompt", "detect_language", "is_action_query"]


