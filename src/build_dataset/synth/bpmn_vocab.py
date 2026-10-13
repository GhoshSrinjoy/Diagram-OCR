"""Business vocabulary for synthetic BPMN diagrams.

Each domain lists the organisations (pools), roles (lanes), task phrases, event names, branch
conditions, messages, data objects and stores of one kind of business process. Task names are
either taken verbatim or composed as verb + object (+ qualifier), so every diagram reads like a
plausible process. Text is written in lower case; `cased()` applies the diagram's casing style.
"""

from __future__ import annotations

import random
import re

# --------------------------------------------------------------------------- #
# Domains
# --------------------------------------------------------------------------- #
DOMAINS: list[dict] = [
    {
        "name": "order",
        "pools": ["Online Shop", "Webshop", "Retailer", "Sales Department", "Shop GmbH", "Acme Corp",
                  "Mail Order Company", "Seller"],
        "partners": ["Customer", "Supplier", "Carrier", "Payment Provider", "Wholesaler",
                     "Logistics Provider", "Buyer", "Manufacturer"],
        "roles": ["Sales", "Warehouse", "Shipping", "Accounting", "Customer Service", "Order Desk",
                  "Packaging", "Finance", "Sales Clerk", "Warehouse Worker", "Back Office"],
        "objects": ["order", "invoice", "payment", "goods", "shipment", "package", "delivery note",
                    "order confirmation", "stock", "parcel", "return", "refund"],
        "tasks": ["check order", "confirm order", "check stock", "pack goods", "ship goods",
                  "send invoice", "receive payment", "archive order", "reject order",
                  "order missing items", "prepare shipment", "issue invoice", "update inventory",
                  "notify customer", "check payment", "send reminder", "cancel order",
                  "print shipping label", "pick items", "hand over to carrier", "process return",
                  "refund money", "check if order is complete", "send order confirmation",
                  "enter order into system", "contact supplier", "reserve items",
                  "calculate shipping costs", "book payment", "check address"],
        "start": ["order received", "new order", "order placed", "return received",
                  "customer order arrived", "order request received"],
        "end": ["order completed", "order shipped", "order rejected", "order cancelled",
                "goods delivered", "order fulfilled", "return processed", "order closed",
                "payment booked"],
        "catch": ["payment received", "goods received", "confirmation received", "reply received",
                  "delivery received", "order changed", "cancellation received"],
        "throw": ["invoice sent", "order confirmed", "customer notified", "reminder sent",
                  "shipment sent", "request sent"],
        "conds": [("in stock", "not in stock"), ("paid", "not paid"), ("accepted", "rejected"),
                  ("complete", "incomplete"), ("available", "not available"),
                  ("standard", "express", "pickup"), ("by phone", "by email", "online"),
                  ("amount > 500", "amount <= 500")],
        "messages": ["order", "invoice", "payment", "order confirmation", "delivery", "reminder",
                     "cancellation", "goods", "shipping notice", "refund"],
        "data": ["order", "invoice", "delivery note", "order list", "packing list", "receipt",
                 "shipping label", "customer data"],
        "stores": ["order database", "ERP system", "inventory DB", "customer database",
                   "product catalogue", "warehouse system"],
    },
    {
        "name": "claim",
        "pools": ["Insurer", "Insurance Company", "Insurance", "Claims Department", "Assurance AG"],
        "partners": ["Customer", "Claimant", "Garage", "Expert", "Policy Holder", "Repair Shop",
                     "Lawyer", "Hospital"],
        "roles": ["Claims Officer", "Senior Claims Officer", "Clerk", "Assessor", "Finance",
                  "Claims Handler", "Notification Department", "Handling Department",
                  "Fraud Team", "Customer Service"],
        "objects": ["claim", "policy", "damage report", "compensation", "documents", "assessment",
                    "settlement", "invoice", "claim form"],
        "tasks": ["register claim", "check policy", "assess damage", "approve claim", "reject claim",
                  "pay compensation", "request documents", "evaluate severity of claim",
                  "send rejection letter", "examine claim", "check if claimant is insured",
                  "write recommendation", "check recommendation", "perform assessment",
                  "schedule payment", "authorize repairs", "check documentation",
                  "send relevant forms", "check completeness", "inform customer",
                  "calculate compensation", "check for fraud", "close claim", "archive claim",
                  "contact expert", "send claim forms"],
        "start": ["claim received", "damage reported", "claim submitted", "new claim",
                  "accident reported", "documents received"],
        "end": ["claim settled", "claim rejected", "claim closed", "payment made",
                "claim processed", "compensation paid", "claim registered"],
        "catch": ["documents received", "forms received", "report received", "invoice received",
                  "reply received", "expertise received"],
        "throw": ["rejection sent", "forms sent", "customer informed", "payment ordered",
                  "repairs authorized", "letter sent"],
        "conds": [("covered", "not covered"), ("simple claim", "complex claim"),
                  ("approved", "rejected"), ("complete", "not complete"), ("insured", "not insured"),
                  ("ok", "not ok"), ("fraud suspected", "no fraud"), ("damage < 1000", "damage >= 1000")],
        "messages": ["claim", "claim forms", "rejection notification", "documents", "compensation",
                     "expertise", "invoice", "update notification", "payment", "report"],
        "data": ["claim", "claim form", "damage report", "policy", "settlement recommendation",
                 "assessment report", "invoice", "documents"],
        "stores": ["claims management system", "SAP system", "customer DB", "policy database",
                   "archive"],
    },
    {
        "name": "hiring",
        "pools": ["Company", "HR Department", "Employer", "Recruiting Agency", "Tech Startup"],
        "partners": ["Applicant", "Candidate", "Job Portal", "Recruiter", "Employee"],
        "roles": ["HR", "Hiring Manager", "Team Lead", "Recruiter", "HR Assistant", "Management",
                  "Department Head", "IT Department", "Works Council"],
        "objects": ["application", "job ad", "interview", "offer", "contract", "references", "CV",
                    "onboarding plan", "position"],
        "tasks": ["screen application", "schedule interview", "conduct interview", "send offer",
                  "sign contract", "reject candidate", "post job ad", "check references",
                  "prepare onboarding", "review CV", "invite candidate", "send rejection",
                  "negotiate salary", "prepare contract", "set up workplace", "create account",
                  "evaluate candidates", "approve hiring", "define requirements",
                  "collect feedback", "send welcome email", "order laptop"],
        "start": ["application received", "position vacant", "new hire needed",
                  "job requested", "vacancy reported"],
        "end": ["position filled", "candidate rejected", "employee onboarded", "contract signed",
                "hiring cancelled", "candidate hired"],
        "catch": ["offer accepted", "documents received", "signed contract received",
                  "answer received", "feedback received"],
        "throw": ["offer sent", "candidate invited", "rejection sent", "contract sent"],
        "conds": [("suitable", "not suitable"), ("accepted", "declined"), ("approved", "rejected"),
                  ("yes", "no"), ("internal", "external"), ("junior", "senior")],
        "messages": ["application", "invitation", "offer", "contract", "rejection", "feedback",
                     "signed contract", "documents"],
        "data": ["application", "CV", "job description", "contract", "interview notes",
                 "offer letter", "candidate list"],
        "stores": ["HR system", "applicant database", "personnel files", "ERP"],
    },
    {
        "name": "loan",
        "pools": ["Bank", "Credit Institute", "Savings Bank", "Lender", "Branch Office"],
        "partners": ["Customer", "Credit Bureau", "Borrower", "Notary", "Applicant", "Guarantor"],
        "roles": ["Loan Officer", "Risk Analyst", "Branch Manager", "Back Office", "Clerk",
                  "Credit Department", "Account Manager", "Compliance"],
        "objects": ["loan", "credit", "application", "risk", "contract", "account", "funds",
                    "credit history", "collateral"],
        "tasks": ["check credit history", "assess risk", "calculate interest", "approve loan",
                  "prepare contract", "disburse funds", "open account", "verify identity",
                  "assess simple risk", "assess advanced risk", "organize disbursement",
                  "reject application", "evaluate application", "request credit report",
                  "check collateral", "sign contract", "send bank card", "send PIN number",
                  "create new bank account", "send welcome pack", "notify customer by email",
                  "check income", "calculate rate"],
        "start": ["loan requested", "new credit requested", "application received",
                  "new online application received", "account requested"],
        "end": ["loan approved", "loan rejected", "contract signed", "funds disbursed",
                "application rejected", "new bank account created", "credit granted"],
        "catch": ["credit report received", "documents received", "signed contract received",
                  "reply received"],
        "throw": ["offer sent", "rejection sent", "customer informed", "report requested"],
        "conds": [("low risk", "high risk"), ("approved", "declined"),
                  ("risk under threshold", "risk above threshold"), ("application approved",
                                                                    "application rejected"),
                  ("ok", "not ok"), ("score >= 600", "score < 600")],
        "messages": ["application", "credit report", "contract", "offer", "rejection",
                     "documents", "bank card", "PIN"],
        "data": ["application", "credit report", "loan contract", "risk assessment",
                 "customer file", "income statement"],
        "stores": ["core banking system", "customer database", "SCHUFA DB", "risk database"],
    },
    {
        "name": "it",
        "pools": ["IT Department", "Service Desk", "IT Services", "Software Company", "Help Desk"],
        "partners": ["User", "Customer", "Vendor", "External Support", "Employee"],
        "roles": ["1st Level Support", "2nd Level Support", "Developer", "Service Desk",
                  "Operations", "Tester", "Product Owner", "Administrator", "Release Manager"],
        "objects": ["ticket", "issue", "incident", "change request", "fix", "release", "bug",
                    "patch", "access"],
        "tasks": ["log ticket", "classify issue", "analyse problem", "provide solution",
                  "escalate ticket", "deploy fix", "close ticket", "inform user", "reproduce bug",
                  "write test", "review code", "implement fix", "test fix", "plan release",
                  "grant access", "reset password", "install software", "update documentation",
                  "check known errors", "assign ticket", "restart server", "monitor system"],
        "start": ["ticket created", "incident reported", "bug reported", "change requested",
                  "access requested", "alarm raised"],
        "end": ["issue resolved", "ticket closed", "fix deployed", "request rejected",
                "release published", "access granted"],
        "catch": ["user feedback received", "approval received", "answer received",
                  "logs received"],
        "throw": ["user notified", "ticket escalated", "release announced", "request sent"],
        "conds": [("solved", "not solved"), ("known issue", "new issue"), ("tests passed",
                                                                            "tests failed"),
                  ("approved", "rejected"), ("yes", "no"), ("critical", "minor")],
        "messages": ["ticket", "feedback", "solution", "request", "logs", "approval", "fix"],
        "data": ["ticket", "error log", "test report", "change request", "release notes"],
        "stores": ["ticket system", "knowledge base", "CMDB", "code repository"],
    },
    {
        "name": "procurement",
        "pools": ["Purchasing", "Procurement Department", "Company", "Buyer Ltd", "Factory"],
        "partners": ["Supplier", "Vendor", "Manufacturer", "Distributor", "Service Provider"],
        "roles": ["Requester", "Purchasing", "Manager", "Accounts Payable", "Goods Receipt",
                  "Buyer", "Controlling", "Department Head", "Warehouse"],
        "objects": ["purchase request", "purchase order", "quote", "goods", "invoice", "supplier",
                    "delivery", "offer"],
        "tasks": ["create purchase request", "approve request", "select supplier",
                  "send purchase order", "receive goods", "check delivery", "pay invoice",
                  "request quotes", "compare offers", "negotiate price", "check budget",
                  "return goods", "book invoice", "check invoice", "release payment",
                  "file complaint", "update supplier list", "sign order"],
        "start": ["demand identified", "material needed", "request submitted",
                  "stock below minimum"],
        "end": ["goods delivered", "invoice paid", "request rejected", "order completed",
                "purchase cancelled"],
        "catch": ["quotes received", "goods received", "invoice received", "order confirmed",
                  "offer received"],
        "throw": ["order sent", "request for quote sent", "complaint sent", "payment sent"],
        "conds": [("approved", "rejected"), ("within budget", "over budget"),
                  ("ok", "damaged"), ("correct", "incorrect"), ("yes", "no"),
                  ("amount > 1000", "amount <= 1000")],
        "messages": ["purchase order", "quote", "invoice", "goods", "payment", "complaint",
                     "order confirmation", "request for quote"],
        "data": ["purchase request", "purchase order", "quote", "invoice", "delivery note",
                 "supplier list"],
        "stores": ["ERP system", "supplier database", "SAP", "budget plan"],
    },
    {
        "name": "hospital",
        "pools": ["Hospital", "Clinic", "Medical Practice", "Emergency Room", "Health Center"],
        "partners": ["Patient", "Insurance", "Lab", "Pharmacy", "General Practitioner",
                     "External Expert"],
        "roles": ["Reception", "Nurse", "Doctor", "Lab", "Billing", "Physician", "Radiology",
                  "QA Department", "Managing Director", "Ward"],
        "objects": ["patient", "sample", "results", "prescription", "appointment", "report",
                    "medication", "referral"],
        "tasks": ["register patient", "measure vital signs", "examine patient", "order lab test",
                  "analyse sample", "prescribe medication", "schedule follow-up",
                  "discharge patient", "take blood sample", "make diagnosis", "admit patient",
                  "create report", "file results", "interview patient", "ask expert",
                  "forward expertise", "fill cheque", "prepare surgery", "inform relatives",
                  "update patient record", "bill treatment"],
        "start": ["patient arrived", "emergency call received", "appointment requested",
                  "rating started", "referral received"],
        "end": ["patient discharged", "treatment completed", "patient admitted", "doctor rated",
                "report sent"],
        "catch": ["results received", "review received", "expertise received",
                  "approval received"],
        "throw": ["results sent", "patient informed", "report sent", "request sent"],
        "conds": [("urgent", "not urgent"), ("normal", "abnormal"), ("yes", "no"),
                  ("insured", "not insured"), ("stable", "critical")],
        "messages": ["referral", "lab results", "invoice", "prescription", "expertise", "cheque",
                     "report", "appointment"],
        "data": ["patient record", "lab results", "prescription", "medical report", "referral"],
        "stores": ["patient database", "hospital information system", "archive"],
    },
    {
        "name": "travel",
        "pools": ["Travel Agency", "Company", "Travel Office", "Tour Operator"],
        "partners": ["Airline", "Hotel", "Customer", "Traveler", "Car Rental"],
        "roles": ["Travel Agent", "Employee", "Manager", "Travel Office", "Accounting",
                  "Booking Team"],
        "objects": ["trip", "flight", "hotel", "booking", "ticket", "travel request",
                    "expense report", "visa"],
        "tasks": ["search flights", "book hotel", "approve travel request", "issue ticket",
                  "submit expense report", "reimburse costs", "book flight", "check availability",
                  "send itinerary", "cancel booking", "rebook flight", "apply for visa",
                  "check receipts", "plan trip"],
        "start": ["travel request received", "trip planned", "booking requested"],
        "end": ["booking confirmed", "trip completed", "costs reimbursed", "request rejected",
                "booking cancelled"],
        "catch": ["confirmation received", "payment received", "ticket received"],
        "throw": ["itinerary sent", "booking request sent", "traveler informed"],
        "conds": [("available", "not available"), ("approved", "rejected"), ("domestic",
                                                                              "international"),
                  ("yes", "no")],
        "messages": ["booking request", "confirmation", "ticket", "invoice", "itinerary",
                     "payment"],
        "data": ["travel request", "itinerary", "ticket", "expense report", "receipts"],
        "stores": ["booking system", "travel database", "expense tool"],
    },
    {
        "name": "university",
        "pools": ["University", "School", "Admissions Office", "Faculty", "College"],
        "partners": ["Student", "Applicant", "Teacher", "Examination Board", "Parents"],
        "roles": ["Admissions Office", "Academic Committee", "Student Office", "Teacher",
                  "Secretary", "Examination Office", "Professor", "Admission Information System"],
        "objects": ["application", "grades", "documents", "enrolment", "exam", "course", "thesis",
                    "certificate"],
        "tasks": ["check documents", "evaluate application", "send admission letter",
                  "enroll student", "assign courses", "check grades", "examine applications",
                  "notify admission office about outcome", "record in system", "generate PDF",
                  "offer elective courses", "collect descriptions", "enter into CBS",
                  "send list to teacher", "choose students", "rebook", "grade exam",
                  "assign students to free courses", "issue certificate", "schedule exam",
                  "check outcome", "mail to committee"],
        "start": ["application received", "semester started", "documents received",
                  "registration opened"],
        "end": ["student enrolled", "application rejected", "admission sent", "exam graded",
                "certificate issued"],
        "catch": ["documents received", "grades received", "answer received", "fee received"],
        "throw": ["admission sent", "rejection sent", "list sent", "invitation sent"],
        "conds": [("sufficient grades", "insufficient grades"), ("admission", "rejection"),
                  ("too many students", "enough places"), ("passed", "failed"), ("yes", "no")],
        "messages": ["application", "documents", "admission", "rejection", "grades",
                     "list of students"],
        "data": ["application", "list of students", "transcript", "course list", "certificate"],
        "stores": ["campus system", "student database", "CBS"],
    },
    {
        "name": "restaurant",
        "pools": ["Restaurant", "Pizzeria", "Bakery", "Catering Service", "Diner"],
        "partners": ["Guest", "Customer", "Delivery Service", "Supplier", "Lieferando"],
        "roles": ["Waiter", "Kitchen", "Cashier", "Delivery", "Chef", "Front Desk", "Driver"],
        "objects": ["order", "meal", "table", "bill", "order note", "drinks", "ingredients"],
        "tasks": ["take order", "prepare meal", "serve food", "print out order", "deliver food",
                  "collect payment", "take order by phone", "receive order by website",
                  "write down order note", "attach order note to pinboard", "confirm order",
                  "bake pizza", "pack order", "assign table", "bring bill", "clean table",
                  "order ingredients", "check reservation"],
        "start": ["order received", "guest arrived", "hungry", "call received", "reservation made"],
        "end": ["order delivered", "guest served", "meal served", "order fulfilled", "bill paid"],
        "catch": ["payment received", "order received", "delivery arrived"],
        "throw": ["order confirmed", "driver informed", "guest notified"],
        "conds": [("by phone", "by website", "on site"), ("available", "sold out"), ("yes", "no"),
                  ("cash", "card")],
        "messages": ["order", "bill", "payment", "confirmation", "meal"],
        "data": ["order", "order note", "bill", "menu", "reservation list"],
        "stores": ["POS system", "order database"],
    },
    {
        "name": "production",
        "pools": ["Factory", "Production Plant", "Manufacturer", "Workshop", "Assembly Plant"],
        "partners": ["Supplier", "Customer", "Logistics", "Quality Lab"],
        "roles": ["Planning", "Production", "Quality Control", "Logistics", "Maintenance",
                  "Assembly", "Engineering", "Shift Lead"],
        "objects": ["production order", "parts", "batch", "material", "product", "machine",
                    "test report"],
        "tasks": ["plan production", "assemble parts", "inspect quality", "rework part",
                  "store product", "order material", "set up machine", "run test",
                  "repair machine", "package product", "label batch", "release batch",
                  "scrap part", "calibrate machine", "document results", "check material"],
        "start": ["production order received", "batch planned", "material arrived",
                  "machine failure"],
        "end": ["batch finished", "product stored", "batch scrapped", "machine repaired",
                "product shipped"],
        "catch": ["material received", "test results received", "spare part received"],
        "throw": ["order sent", "batch released", "supplier notified"],
        "conds": [("ok", "not ok"), ("passed", "failed"), ("yes", "no"), ("in tolerance",
                                                                          "out of tolerance")],
        "messages": ["material", "production order", "test results", "order", "spare part"],
        "data": ["production plan", "test report", "bill of materials", "work order"],
        "stores": ["MES", "ERP system", "quality database"],
    },
    {
        "name": "complaint",
        "pools": ["Service Center", "Company", "Customer Care", "Electronics Store"],
        "partners": ["Customer", "Manufacturer", "Field Service", "Courier"],
        "roles": ["Service Center", "Quality Management", "Field Service", "Customer Care",
                  "Technician", "Team Lead"],
        "objects": ["complaint", "product", "replacement", "compensation", "refund", "voucher"],
        "tasks": ["record complaint", "investigate cause", "offer compensation", "replace product",
                  "send apology", "check warranty", "repair product", "send voucher",
                  "close complaint", "contact customer", "pick up product", "analyse defect",
                  "send replacement", "issue refund"],
        "start": ["complaint received", "defect reported", "return requested"],
        "end": ["complaint closed", "product replaced", "refund issued", "complaint rejected"],
        "catch": ["product received", "customer reply received", "report received"],
        "throw": ["apology sent", "voucher sent", "customer informed"],
        "conds": [("justified", "not justified"), ("under warranty", "no warranty"), ("yes", "no"),
                  ("repairable", "not repairable")],
        "messages": ["complaint", "product", "replacement", "voucher", "refund", "report"],
        "data": ["complaint form", "defect report", "warranty card", "receipt"],
        "stores": ["CRM", "complaint database"],
    },
    {
        "name": "permit",
        "pools": ["City Council", "Municipality", "Tribunal", "Building Authority", "Tax Office",
                  "Public Office"],
        "partners": ["Applicant", "Citizen", "Owner", "Architect", "Neighbour", "Court"],
        "roles": ["Front Office", "Case Worker", "Department Head", "Cashier", "Tribunal",
                  "Inspector", "Legal Department"],
        "objects": ["application", "permit", "fee", "hearing", "documents", "request", "decision"],
        "tasks": ["check application", "request missing documents", "issue permit", "collect fee",
                  "schedule hearing", "set hearing date", "retrieve files",
                  "check if request is acceptable", "check if request is compliant",
                  "allocate hearing date", "inspect site", "send decision", "archive files",
                  "calculate fee", "publish decision", "consult neighbours"],
        "start": ["application received", "request for compensation received",
                  "request submitted"],
        "end": ["permit issued", "process completed", "application rejected", "decision sent"],
        "catch": ["fee received", "documents received", "objection received"],
        "throw": ["fees notice sent", "decision sent", "applicant informed"],
        "conds": [("fee paid", "fee not paid"), ("complete", "incomplete"), ("approved",
                                                                            "rejected"),
                  ("yes", "no"), ("acceptable", "not acceptable")],
        "messages": ["compensation request", "fee", "fees notice", "decision", "documents",
                     "objection"],
        "data": ["files", "compensation request", "fees notice", "permit", "application form"],
        "stores": ["case management system", "land register", "archive"],
    },
    {
        "name": "supplier",
        "pools": ["Supplier", "Wholesaler", "Distributor", "Producer"],
        "partners": ["Retailer", "Customer", "Shop", "Buyer"],
        "roles": ["Sales", "Order Management", "Logistics", "Billing", "Key Account"],
        "objects": ["offer", "order", "order change", "confirmation", "invoice"],
        "tasks": ["prepare offer", "update offer", "process order normally", "update order",
                  "cancel order", "send offer", "check order change", "confirm change",
                  "ship order", "create invoice", "process order", "check capacity"],
        "start": ["offer request received", "request received", "order received"],
        "end": ["offer processed", "order cancelled", "order processed", "order shipped"],
        "catch": ["order confirmation received", "order change received",
                  "order cancellation received", "response received"],
        "throw": ["offer sent", "offer sent to retailer", "retailer asked for confirmation",
                  "invoice sent"],
        "conds": [("order changed three times", "less than three changes"), ("yes", "no"),
                  ("3rd time", "1st or 2nd time"), ("accepted", "declined")],
        "messages": ["offer request", "offer", "order confirmation", "order change",
                     "order cancellation", "confirmation request", "response"],
        "data": ["offer", "order", "price list"],
        "stores": ["CRM", "order system"],
    },
]

# --------------------------------------------------------------------------- #
# Generic parts used across domains
# --------------------------------------------------------------------------- #
VERBS = [
    "check", "review", "verify", "approve", "reject", "send", "receive", "prepare", "create",
    "update", "register", "archive", "notify", "inform", "calculate", "schedule", "request",
    "forward", "assign", "process", "evaluate", "analyse", "confirm", "sign", "record", "file",
    "print", "enter", "validate", "submit", "cancel", "complete", "close", "inspect", "document",
    "publish", "book", "contact", "escalate", "monitor", "plan", "select", "compare", "issue",
    "revise", "collect", "deliver", "store", "scan", "correct", "discuss", "clarify", "release",
    "determine", "decide on", "fill in", "look up", "hand over", "mark",
]
QUALIFIERS = [
    "by email", "in system", "for approval", "to customer", "with supplier", "manually",
    "online", "by phone", "in ERP", "to manager", "again", "in detail", "for errors",
    "with team", "by post", "in database",
]
TIMERS = [
    "2 days", "24 hours", "1 week", "every Monday", "end of month", "after 3 days",
    "deadline reached", "30 min", "every morning", "Friday 5 pm", "48 hours", "14 days",
    "one week before start of school year", "few days later", "every month", "wait 1 hour",
    "10 working days", "daily at 8:00", "after 2 weeks", "1st of month", "5 days",
    "time out", "3 days passed", "one day before",
]
COND_GENERIC = [
    ("yes", "no"), ("ok", "not ok"), ("approved", "rejected"), ("complete", "incomplete"),
    ("valid", "invalid"), ("accepted", "declined"), ("found", "not found"), ("correct", "wrong"),
    ("positive", "negative"), ("high", "low"), ("low", "medium", "high"),
    ("small", "medium", "large"), ("A", "B", "C"), ("true", "false"),
]
LOOP_BACK = ["no", "not ok", "incomplete", "retry", "rework needed", "not approved", "invalid",
             "not complete", "again", "errors found", "not correct", "rejected"]
LOOP_FWD = ["yes", "ok", "complete", "approved", "valid", "done", "correct", "accepted"]
STATES = ["approved", "checked", "signed", "registered", "paid", "completed", "draft",
          "rejected", "updated", "examined", "marked", "new", "final", "sent", "archived"]
NOTES = [
    "within 2 days", "for each item", "manual step", "max. 3 attempts", "by email",
    "only if amount > 500 EUR", "done by the team lead", "see checklist", "via online portal",
    "every Monday", "for each teacher", "until respond in time", "4-eyes principle",
    "automated by system", "customer gets a copy", "repeat for every order",
    "takes about 1 hour", "use template A", "if necessary", "legal deadline: 30 days",
    "only for new customers", "send via registered mail", "1st, 2nd or 3rd time",
    "in parallel possible", "result is stored in DB", "check with manager first",
    "SLA: 4 hours", "via phone or email", "approx. 15 min", "copy to archive",
]
GW_QUESTIONS = ["{o} ok?", "{o} approved?", "{o} complete?", "{o} valid?", "{o} available?",
                "{o} correct?", "accept {o}?", "check {o}?", "{o} paid?", "type of {o}?"]

HANDWRITTEN_SAFE = re.compile(r"[^A-Za-z0-9 .,:;!?'\"()\[\]/&+<>=%#*-]")


def pick_domain(rng: random.Random) -> dict:
    return rng.choice(DOMAINS)


def cased(text: str, mode: str) -> str:
    """mode: lower (as written), sentence, title, upper."""
    if mode == "sentence":
        return text[:1].upper() + text[1:]
    if mode == "title":
        return " ".join(w[:1].upper() + w[1:] if w[:1].isalpha() else w for w in text.split(" "))
    if mode == "upper":
        return text.upper()
    return text


def task_name(rng: random.Random, dom: dict) -> str:
    r = rng.random()
    if r < 0.6:
        return rng.choice(dom["tasks"])
    obj = rng.choice(dom["objects"])
    verb = rng.choice(VERBS)
    name = f"{verb} {obj}"
    if r > 0.9:
        name += " " + rng.choice(QUALIFIERS)
    return name


def event_name(rng: random.Random, dom: dict, kind: str) -> str:
    """kind: start, end, catch, throw, timer."""
    if kind == "timer":
        return rng.choice(TIMERS)
    if rng.random() < 0.75:
        return rng.choice(dom[kind])
    obj = rng.choice(dom["objects"])
    pattern = {
        "start": ["{o} received", "new {o}", "{o} requested", "{o} arrived", "need for {o}"],
        "end": ["{o} completed", "{o} processed", "{o} closed", "{o} rejected", "{o} sent",
                "{o} archived", "{o} done"],
        "catch": ["{o} received", "{o} arrived", "reply to {o} received"],
        "throw": ["{o} sent", "{o} forwarded", "send {o}"],
    }[kind]
    return rng.choice(pattern).format(o=obj)


def conditions(rng: random.Random, dom: dict, n: int) -> list[str]:
    pool = [c for c in dom["conds"] if len(c) == n] + [c for c in COND_GENERIC if len(c) == n]
    if not pool:
        pool = [c for c in COND_GENERIC if len(c) >= n]
        return list(rng.choice(pool)[:n])
    out = list(rng.choice(pool))
    if rng.random() < 0.5:
        out.reverse()
    return out


def data_name(rng: random.Random, dom: dict) -> str:
    name = rng.choice(dom["data"])
    if rng.random() < 0.3:
        name += " [" + rng.choice(STATES) + "]"
    return name


def typo(text: str, rng: random.Random) -> str:
    """One handwriting slip: drop, double or swap a letter in a longer word."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    op = rng.random()
    if op < 0.4:
        w = w[:j] + w[j + 1:]
    elif op < 0.7:
        w = w[:j] + w[j] + w[j:]
    else:
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    words[i] = w
    return " ".join(words)
