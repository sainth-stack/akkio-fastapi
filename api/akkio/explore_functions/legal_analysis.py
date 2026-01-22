import json
from typing import Optional
import pandas as pd
from .llm import get_openai_client, get_language_context, call_llm_with_usage



def analyze_legal_content(df: pd.DataFrame, query: str, email: str = None) -> str:

    """
    Analyze legal document content and provide professional structured response with reference links
    """
    try:
        document_text = " ".join(df['text_content'].astype(str).tolist())
        doc_type = df['document_type'].iloc[0] if 'document_type' in df.columns else 'Unknown'
        is_legal = df['is_legal_document'].iloc[0] if 'is_legal_document' in df.columns else False
        is_contract = df['is_contract'].iloc[0] if 'is_contract' in df.columns else False
        is_policy = df['is_policy'].iloc[0] if 'is_policy' in df.columns else False
        language, language_instructions = get_language_context(query)
        if language == "arabic":
            html_template = """
            <div dir="rtl" lang="ar">
            <h3>نظام التحليل القانوني</h3>
            <h4>الإجابة المباشرة</h4>
            <p>بناءً على الوثائق القانونية لدولة الإمارات العربية المتحدة، يقدم التحليل الشامل التالي معالجة للمسؤوليات والالتزامات القانونية للمحامين في دولة الإمارات العربية المتحدة، مع التركيز بشكل خاص على تمثيل العملاء، تضارب المصالح، والسلوك المهني.</p>
            <p>[نظرة شاملة على محتوى الوثيقة القانونية والنتائج الرئيسية المتعلقة بسؤال المستخدم]</p>
            <h4>المواضيع القانونية الرئيسية</h4>
            <ul>
            <li><strong>[الموضوع القانوني الأول]:</strong> [شرح مفصل مع مراجع محددة لمحتوى الوثيقة القانونية]</li>
            <li><strong>[الموضوع القانوني الثاني]:</strong> [شرح مفصل مع مراجع محددة لمحتوى الوثيقة القانونية]</li>
            <li><strong>[الموضوع القانوني الثالث]:</strong> [شرح مفصل مع مراجع محددة لمحتوى الوثيقة القانونية]</li>
            </ul>
            <h4>الأحكام القانونية/التنظيمية</h4>
            <ul>
            <li><strong>[اسم الحكم القانوني]:</strong> [تفاصيل محددة وآثار قانونية من الوثيقة]</li>
            <li><strong>[حكم قانوني آخر]:</strong> [تفاصيل محددة وآثار قانونية من الوثيقة]</li>
            </ul>
            <h4>الإرشادات القانونية العملية</h4>
            <ul>
            <li><strong>[مجال الإرشاد القانوني الأول]:</strong> [توصيات قانونية قابلة للتنفيذ بناءً على محتوى الوثيقة]</li>
            <li><strong>[مجال الإرشاد القانوني الثاني]:</strong> [توصيات قانونية قابلة للتنفيذ بناءً على محتوى الوثيقة]</li>
            </ul>
            <h4>الملخص القانوني</h4>
            <p>[ملخص شامل مع النقاط القانونية الرئيسية والخطوات التالية بناءً على التحليل القانوني]</p>
            <hr>
            <p><em>✅ تم إكمال التحليل القانوني المهني</em></p>
            <p><em>هذا التحليل مبني على محتوى الوثيقة القانونية المقدمة وهو للإرشاد فقط. للمسائل القانونية المحددة، يرجى استشارة المحامين المؤهلين.</em></p>
            <h4>المراجع القانونية</h4>
            <p><strong>المصدر:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
            <p><strong>مرجع إضافي:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
            </div>
            """
        else:
            html_template = """
            <h3>LEGAL ANALYSIS SYSTEM</h3>
            <h4>Direct Answer</h4>
            <p>Based on the UAE legal documentation, the following comprehensive analysis addresses the legal responsibilities and obligations of lawyers in the UAE, particularly focusing on client representation, conflict of interest, and professional conduct.</p>
            <p>[Comprehensive overview of the legal document content and key findings related to the user's query]</p>
            <h4>Key Legal Topics Covered</h4>
            <ul>
            <li><strong>[Legal Topic 1]:</strong> [Detailed explanation with specific references to legal document content]</li>
            <li><strong>[Legal Topic 2]:</strong> [Detailed explanation with specific references to legal document content]</li>
            <li><strong>[Legal Topic 3]:</strong> [Detailed explanation with specific references to legal document content]</li>
            </ul>
            <h4>Legal/Regulatory Provisions</h4>
            <ul>
            <li><strong>[Legal Provision Name]:</strong> [Specific details and legal implications from the document]</li>
            <li><strong>[Another Legal Provision]:</strong> [Specific details and legal implications from the document]</li>
            </ul>
            <h4>Practical Legal Guidance</h4>
            <ul>
            <li><strong>[Legal Guidance Area 1]:</strong> [Actionable legal recommendations based on document content]</li>
            <li><strong>[Legal Guidance Area 2]:</strong> [Actionable legal recommendations based on document content]</li>
            </ul>
            <h4>Legal Summary</h4>
            <p>[Comprehensive summary with key legal takeaways and next steps based on the legal analysis]</p>
            <hr>
            <p><em>✅ PROFESSIONAL LEGAL ANALYSIS COMPLETED</em></p>
            <p><em>This legal analysis is based on the provided legal document content and is for guidance only. For specific legal matters, please consult with qualified legal professionals.</em></p>
            <h4>Legal References</h4>
            <p><strong>SOURCE:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
            <p><strong>Additional Reference:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
            """
        analysis_prompt = f"""
        You are a senior legal analysis expert specializing in UAE legal documents and regulations. Analyze the following {doc_type} legal document content and provide a comprehensive, professional legal analysis.
        {language_instructions}
        LEGAL DOCUMENT METADATA:
        - Document Type: {doc_type}
        - Legal Document: {is_legal}
        - Contract Document: {is_contract}
        - Policy Document: {is_policy}
        - Total Lines: {len(df)}
        - Language: {language.upper()}
        LEGAL DOCUMENT CONTENT:
        {document_text[:8000]}
        USER QUERY: {query}
        Provide your legal analysis in the EXACT HTML format below:
        {html_template}
        IMPORTANT: Return ONLY the HTML content above, no additional text or explanations.
        """
        response = call_llm_with_usage(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior legal analysis expert specializing in UAE legal documents and regulations. Provide comprehensive, professional legal analysis in the exact HTML format requested."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=2000,
            temperature=0.3,
            email=email
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"""
        <h3>LEGAL ANALYSIS SYSTEM</h3>
        <h4>Legal Analysis Error</h4>
        <p>There was an error analyzing the legal document: {str(e)}</p>
        <p>Please ensure the legal document is properly formatted and try again.</p>
        <h4>Legal References</h4>
        <p><strong>SOURCE:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
        <p><strong>Additional Reference:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
        """








