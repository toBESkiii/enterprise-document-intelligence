"""Streamlit interface using Neon search and Groq-hosted source-based answers."""

from uuid import uuid4

import streamlit as st


st.set_page_config(page_title="Unilever Document Intelligence", layout="wide")


@st.cache_resource(show_spinner=False)
def load_engine():
    # Load cloud-connected search and answering only when a question is submitted.
    from document_qa_groq import GroqDocumentQA
    return GroqDocumentQA()


# These names must match the document names stored in PostgreSQL.
DOCUMENT_FILES = {
    "UL01": "UL01_Annual_Report_2025.pdf",
    "UL02": "UL02_Annual_Report_2024.pdf",
    "UL03": "UL03_Climate_Transition_Action_Plan_2024.pdf",
    "UL04": "UL04_Modern_Slavery_Statement_2026.pdf",
    "UL05": "UL05_Modern_Slavery_Statement_2025.pdf",
    "UL06": "UL06_Responsible_Partner_Policy.pdf",
    "UL07": "UL07_Responsible_Partner_Implementation_Guidance.pdf",
    "UL08": "UL08_Audit_Requirements.pdf",
    "UL09": "UL09_People_and_Nature_Policy.pdf",
    "UL10": "UL10_Code_of_Business_Principles.pdf",
}

PREPARED_OVERVIEW_NOTE = (
    "Prepared overview checked against selected saved passages; see the PDF page "
    "references. Not generated live or an exhaustive summary. Check the source "
    "document for full context."
)

# Prepared display text, suggested questions and PDF page counts.
# Opening an overview does not call Groq or search the database.
DOCUMENTS = {
    "UL01": {
        "title": "Annual Report 2025",
        "pages": 285,
        "summary": (
            "Unilever's 2025 annual report brings together its strategy, business performance, "
            "governance, financial statements and sustainability disclosures (PDF p. 3). "
            "The business overview describes its four continuing Business Groups and "
            "reports headline turnover and research-and-development spending (PDF p. 5). "
            "Unless otherwise stated, figures are presented on a continuing-operations "
            "basis, with comparative figures re-presented following the Ice Cream "
            "demerger (PDF p. 3). This reporting basis matters when comparing years."
        ),
        "questions": [
            "What was 2025 turnover, and was the Ice Cream business included?",
            "How much did Unilever spend on Research and Development in 2025?",
            "What does the report say about the Supplier Climate Programme?",
        ],
    },
    "UL02": {
        "title": "Annual Report 2024",
        "pages": 305,
        "summary": (
            "Unilever's 2024 annual report describes its Growth Action Plan 2030 and "
            "covers business performance, people, sustainability, risks, governance and "
            "financial statements (PDF p. 4). Its business overview includes five "
            "Business Groups, including Ice Cream, and presents turnover, brand and "
            "marketing investment, and international reach (PDF p. 5). It also reports "
            "research-and-development spending, technology-enabled operations and "
            "workforce indicators (PDF p. 6). These figures belong to the reporting "
            "scope and period stated in this edition."
        ),
        "questions": [
            "What was 2024 turnover and its year-on-year percentage change?",
            "How much was invested in brands and marketing in 2024?",
            "What share of Group turnover came from emerging markets?",
        ],
    },
    "UL03": {
        "title": "Climate Transition Action Plan 2024",
        "pages": 53,
        "summary": (
            "The 2024 Climate Transition Action Plan sets out Unilever's emissions-reduction "
            "plans for its operations and wider value chain (PDF pp. 5, 7). It describes "
            "an ambition to reach net zero across the value chain by 2039 and excludes "
            "purchasing carbon credits to meet its near-term GHG targets (PDF p. 7). "
            "Priority areas include suppliers, product reformulation, agriculture, "
            "packaging and logistics (PDF p. 18). This is a plan with targets and "
            "dependencies; annual progress reporting is published separately (PDF p. 5)."
        ),
        "questions": [
            "By what year does Unilever aim for net zero across its value chain?",
            "How does the Supplier Climate Programme help reduce Scope 3 emissions?",
            "Does the plan include buying carbon credits for near-term GHG targets?",
        ],
    },
    "UL04": {
        "title": "Modern Slavery Statement 2026",
        "pages": 17,
        "summary": (
            "Published in March 2026, this Modern Slavery Statement covers the twelve "
            "months ending 31 December 2025 (PDF pp. 1–2). It describes Unilever's "
            "approach to identifying, preventing and addressing forced labour, alongside "
            "governance, access to remedy and monitoring of effectiveness (PDF pp. 2–3). "
            "It sets out partner-compliance expectations and responsible-recruitment "
            "requirements, including employer payment of recruitment fees and "
            "reimbursement where workers have paid them (PDF p. 6)."
        ),
        "questions": [
            "Who must pay recruitment fees and expenses?",
            "What must happen if workers have already paid recruitment fees?",
            "What happens to new suppliers who do not declare RPP compliance?",
        ],
    },
    "UL05": {
        "title": "Modern Slavery Statement 2025",
        "pages": 16,
        "summary": (
            "Published in March 2025, this Modern Slavery Statement covers the twelve "
            "months ending 31 December 2024 (PDF pp. 1–2). It explains Unilever's "
            "approach to preventing, detecting and responding to slavery in its business "
            "and supply chain (PDF p. 2). It describes partner requirements, voluntary "
            "work and recruitment-fee expectations (PDF p. 5), and discusses corrective "
            "actions, worker repayment, verification and preventing repeated problems "
            "when forced-labour impacts are identified (PDF p. 12)."
        ),
        "questions": [
            "Give two examples of corrective actions after an audit.",
            "How is completion of corrective actions checked?",
            "What should happen when workers have paid recruitment fees?",
        ],
    },
    "UL06": {
        "title": "Responsible Partner Policy",
        "pages": 37,
        "summary": (
            "The 2022 Responsible Partner Policy sets expectations for doing business "
            "with Unilever (PDF p. 3). Its 17 Fundamental Principles sit under Business "
            "Integrity & Ethics, Human Rights, and Planet (PDF pp. 2, 4). It distinguishes "
            "mandatory requirements and management systems from future requirements "
            "and leading practices (PDF p. 4). Partners can use their own policies "
            "and practices, provided they meet or exceed the mandatory requirements "
            "(PDF p. 6); a higher contractual or product-specification standard takes "
            "precedence where applicable (PDF p. 4)."
        ),
        "questions": [
            "What are the three main pillars of the policy?",
            "Can partners use their own policies, and what minimum must they achieve?",
            "Which standard applies if a contract requires a higher standard?",
        ],
    },
    "UL07": {
        "title": "Responsible Partner Implementation Guidance",
        "pages": 112,
        "summary": (
            "This 2023 guidance explains how partners can put the Responsible Partner "
            "Policy into practice through policies, processes and procedures suited to "
            "their business and risks (PDF pp. 1, 3–4). It covers assessing risks, taking "
            "action, engaging workers, tracking performance and communicating, together "
            "with remediation and grievance channels that protect people from "
            "retaliation (PDF pp. 6–7). It should be read alongside the RPP; this edition "
            "covers Principles 1–13 and 15–17, with separate guidance planned for "
            "Principle 14 on Land Rights (PDF p. 3)."
        ),
        "questions": [
            "What due diligence activities does the guidance describe?",
            "Who should have access to internal speak-up channels?",
            "When should third-party due diligence take place?",
        ],
    },
    "UL08": {
        "title": "Audit Requirements",
        "pages": 5,
        "summary": (
            "This May 2025 document explains how risk assessments determine which "
            "Unilever partners need SMETA (4-pillar) audits or EcoVadis assessments "
            "(PDF p. 1). Partners pay for audits and must share results and address "
            "findings (PDF p. 2). It lists approved SMETA auditors and describes "
            "system reporting, corrective actions and score-based audit validity "
            "(PDF pp. 3–4). Key Incident action plans are due within seven days of "
            "the initial audit; remediation and verification are generally due "
            "within 90 days, with longer implementation possible for some "
            "Collaborative Action Required (CAR) findings subject to Unilever "
            "review (PDF p. 4). EcoVadis validity and improvement requirements "
            "depend on assessment scores (PDF p. 5)."
        ),
        "summary_note": (
            "Prepared overview checked against saved text from PDF pages 1–5 "
            "(May 2025 edition). Not generated live; consult the full document for details."
        ),
        "questions": [
            "Which audit standard applies to partners handling or supplying raw materials and finished goods?",
            "Who pays for the audit?",
            "How long is an EcoVadis assessment valid with a score of 65 or above?",
        ],
    },
    "UL09": {
        "title": "People and Nature Policy",
        "pages": 10,
        "summary": (
            "The December 2020 People & Nature Policy sets principles for in-scope "
            "materials: protecting natural ecosystems, respecting human rights, "
            "transparency and traceability, and positive outcomes for people and nature "
            "(PDF pp. 2–3). It describes human-rights due diligence and supply-chain "
            "mapping requirements for Direct Suppliers (PDF pp. 6–7), plus compliance "
            "verification, grievance mechanisms and remediation (PDF p. 9). It should "
            "be read with the separate Policy Guidelines, which define scope, timelines "
            "and technical terms (PDF p. 3)."
        ),
        "questions": [
            "What methods can Direct Suppliers use to maintain a chain of custody?",
            "Which three earlier policies or position papers did this policy replace?",
            "What does the policy say about tracing materials to agricultural origin?",
        ],
    },
    "UL10": {
        "title": "Code of Business Principles",
        "pages": 34,
        "summary": (
            "This January 2026 edition sets out the Code of Business Principles and "
            "detailed Code Policies, which are mandatory for Unilever employees, "
            "management and directors (PDF pp. 2–3). Its themes include conduct, "
            "countering corruption, respecting people, safeguarding information and "
            "external engagement (PDF p. 4). It explains how to report actual or potential "
            "breaches and protect reporters from retaliation (PDF p. 11), and includes "
            "restrictions on trading and sharing inside information (PDF p. 29)."
        ),
        "questions": [
            "Which channels can employees use to report Code breaches?",
            "What must employees not do with securities when they possess inside information?",
            "What does the Code say about sharing inside information?",
        ],
    },
}

COLLECTION_SUMMARY = (
    "These ten Unilever documents bring together business reporting, climate plans "
    "and responsible-business policies. The annual reports cover performance, "
    "financial results, governance and sustainability, while the climate plan "
    "sets out emissions-reduction ambitions and actions. The modern slavery "
    "statements address forced-labour risks and responses. The partner policy, "
    "implementation guidance and audit requirements explain expectations for "
    "business partners. The People & Nature Policy focuses on sourcing, human "
    "rights and traceability, and the Code covers employee conduct. Publication "
    "dates and reporting periods differ, so check each document's date and scope."
)

COLLECTION_OVERVIEW_NOTE = (
    "Prepared collection overview based on the source-referenced document overviews. "
    "Not generated live. These saved editions are not a live view of current policies."
)

COLLECTION_QUESTIONS = [
    DOCUMENTS["UL08"]["questions"][0],
    "How does Unilever's Supplier Climate Programme help suppliers reduce Scope 3 emissions?",
    "What standards must Unilever's business partners meet?",
    "How can employees report breaches of the Code of Business Principles?",
]

# Formatting only: the extra top space keeps the heading below Streamlit's header.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 5rem;
    }
    h1 { color:#162b49; letter-spacing:-0.03em; }
    .collection-heading { color:#245ee0; font-size:14px;
        font-weight:600; letter-spacing:0.08em; text-transform:uppercase; }
    .st-key-suggestions button, .st-key-document_list button {
        text-align:left; justify-content:flex-start; border-radius:10px;
    }
    .st-key-suggestions button p, .st-key-document_list button p {
        text-align:left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state remembers this visitor's choices between button clicks.
for key, default in {
    "edi_document": "ALL",
    "edi_question": "",
    "edi_submission": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Do not keep an old local-model answer after switching the answering backend.
if st.session_state.get("edi_backend_version") != "neon-groq-v1":
    st.session_state.edi_submission = None
    st.session_state.edi_backend_version = "neon-groq-v1"


def select_document(document_id):
    st.session_state.edi_document = document_id
    st.session_state.edi_question = ""
    st.session_state.edi_submission = None
    # Only document navigation requests a scroll, including a repeated selection.
    st.session_state.edi_scroll_request = uuid4().hex


def select_question(question):
    st.session_state.edi_question = question
    st.session_state.edi_submission = None


def scroll_to_top_if_requested():
    """Run once after document navigation, not on normal question/form reruns."""
    request_id = st.session_state.pop("edi_scroll_request", None)
    if request_id is None:
        return

    # Only our generated UUID enters this trusted script. Never insert question,
    # answer, or source text into executable HTML.
    st.html(
        f"""
        <script>
        (() => {{
            const requestId = "{request_id}";
            if (window.__ediScrollRequest === requestId) return;
            window.__ediScrollRequest = requestId;
            let framesRemaining = 60;

            function scrollToOverview() {{
                // A newer navigation supersedes an earlier pending scroll.
                if (window.__ediScrollRequest !== requestId) return;
                const heading = document.getElementById("edi-page-title");
                if (!heading) {{
                    if (--framesRemaining > 0) {{
                        requestAnimationFrame(scrollToOverview);
                    }}
                    return;
                }}

                // Scroll the heading's ancestors, including Streamlit's inner
                // scrolling container, without relying on private CSS classes.
                let container = heading.parentElement;
                while (container) {{
                    if (container.scrollTop || container.scrollLeft) {{
                        container.scrollTo({{top: 0, left: 0, behavior: "instant"}});
                    }}
                    container = container.parentElement;
                }}
                window.scrollTo({{top: 0, left: 0, behavior: "instant"}});

                // Keep keyboard navigation at the new overview as well.
                heading.setAttribute("tabindex", "-1");
                heading.focus({{preventScroll: true}});
            }}

            // Let the new title and overview reach the browser before scrolling.
            requestAnimationFrame(() => requestAnimationFrame(scrollToOverview));
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


# Display either the collection overview or the selected document's overview.
selected_id = st.session_state.edi_document
selected_doc = DOCUMENTS.get(selected_id)
scope = selected_doc["title"] if selected_doc else "Whole collection"
questions = selected_doc["questions"] if selected_doc else COLLECTION_QUESTIONS

st.caption("Enterprise Document Intelligence | Cloud-connected document Q&A")
st.markdown('<p class="collection-heading">Unilever document collection</p>', unsafe_allow_html=True)
st.title(
    selected_doc["title"] if selected_doc else "What would you like to know?",
    anchor="edi-page-title",
)
st.write(selected_doc["summary"] if selected_doc else COLLECTION_SUMMARY)

summary_note = (
    selected_doc.get("summary_note", PREPARED_OVERVIEW_NOTE)
    if selected_doc else COLLECTION_OVERVIEW_NOTE
)
st.caption(summary_note)
st.divider()

left, middle, right = st.columns([1.1, 2.3, 1.3], gap="medium")

with left, st.container(key="suggestions"):
    st.subheader("Suggested questions")
    st.caption("Choose one to fill the question box.")
    for index, question in enumerate(questions):
        st.button(
            question,
            key=f"suggested_{selected_id}_{index}",
            width="stretch",
            on_click=select_question,
            args=(question,),
        )

with middle:
    st.subheader("Ask a question")
    st.caption(f"Question scope: {scope}")
    st.caption(
        "Questions and selected source passages are sent to Groq to generate answers. "
        "Do not enter personal or confidential information. "
        "Free service limits may temporarily prevent answers."
    )

    with st.form("question_form"):
        st.text_area(
            "What would you like to find out?",
            key="edi_question",
            height=160,
            max_chars=2000,
            placeholder="Type a question or choose one on the left...",
        )
        submitted = st.form_submit_button("Ask question", type="primary")

    if submitted:
        question = st.session_state.edi_question.strip()
        st.session_state.edi_submission = None

        if not question:
            st.warning("Please type a question or choose one on the left.")
        else:
            st.session_state.edi_submission = {
                "question": question,
                "document_id": selected_id,
            }

            try:
                document = None if selected_id == "ALL" else DOCUMENT_FILES[selected_id]

                with st.spinner("Loading the search engine if needed...", show_time=True):
                    engine = load_engine()

                with st.spinner("Searching Neon and generating an answer with Groq...", show_time=True):
                    result = engine.ask(question, document)

                st.session_state.edi_submission["result"] = result

            except Exception as error:
                # Do not display passwords, connection details or raw tracebacks.
                st.session_state.edi_submission["error_type"] = type(error).__name__

    submission = st.session_state.edi_submission

    if submission and submission["document_id"] == selected_id:
        st.markdown("**Your submitted question**")
        st.text(submission["question"])

        if "error_type" in submission:
            st.error(
                "The online request could not be completed. Check your internet connection. "
                "If the service's free allowance has been reached, wait for it to reset. "
                "If the problem continues, the app owner should check the Neon/Groq "
                "settings and cached embedding model privately. "
                "No answer was saved for this attempt."
            )
            st.caption(f"Error type: {submission['error_type']}")

        elif "result" in submission:
            result = submission["result"]
            st.markdown("**Answer**")

            if result["refused"]:
                st.info(result["answer"])
                st.caption(
                    "This means the retrieved passages were insufficient; "
                    "it does not prove the answer is absent from every document."
                )
            else:
                st.markdown(result["answer"])

            st.caption(
                f"Search: {result['retrieval_seconds']:.2f}s | "
                f"Answer generation: {result['answer_seconds']:.1f}s | "
                f"Total: {result['total_seconds']:.1f}s (excluding model setup)"
            )

            for warning in result["warnings"]:
                st.warning(warning)

            st.markdown("**Source passages supplied to the model**")
            st.caption(
                f"{len(result['sources'])} supplied from "
                f"{result['retrieved_count']} retrieved passages. "
                "Open a passage to check the answer against its text."
            )

            for source in result["sources"]:
                label = (
                    f"[{source['id']}] {source['document']} "
                    f"— PDF page {source['pdf_page']}"
                )
                with st.expander(label):
                    st.caption(f"Chunk: {source['chunk_id']}")
                    st.text(source["text"])

    else:
        st.info("Choose a suggested question or type your own, then press Ask question.")

    st.caption(
        "Answers can be wrong or incomplete, especially comparisons across reports. "
        "Check source passages, including conditions, dates and exceptions, "
        "before relying on an answer."
    )

# Keep the complete numbered document list visible for every selection.
with right, st.container(key="document_list"):
    st.subheader("Explore documents")
    st.caption(f"Table of contents | {len(DOCUMENTS)} PDFs")
    st.button(
        "Whole collection",
        key="document_ALL",
        type="primary" if selected_id == "ALL" else "secondary",
        width="stretch",
        on_click=select_document,
        args=("ALL",),
    )
    for number, (document_id, document) in enumerate(DOCUMENTS.items(), start=1):
        st.button(
            f"{number:02d}. {document['title']}",
            key=f"document_{document_id}",
            help=f"{document['pages']} PDF pages",
            type="primary" if selected_id == document_id else "secondary",
            width="stretch",
            on_click=select_document,
            args=(document_id,),
        )

st.divider()
st.caption("Portfolio prototype | Neon read-only search + Groq answers | Not affiliated with Unilever")

# Run after the new view has been rendered. The request is consumed just once.
scroll_to_top_if_requested()

