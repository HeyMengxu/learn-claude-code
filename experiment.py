"""
experiment.py
=============
Measures token growth across memory conditions for educational agent study.

Three conditions:
  A) Append-only   — full history concatenated (baseline)
  B) RAG           — top-k semantic retrieval (simple keyword version)
  C) Hierarchical  — 3-tier compression (our approach)

Usage:
  1. Set ANTHROPIC_API_KEY in your .env or environment
  2. Run: python experiment.py
  3. Results saved to results.json and a simple text report
"""

import os
import json
import time
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"  # cheapest model for experiments

# ─────────────────────────────────────────────
# STEP 1: Generate synthetic student sessions
# ─────────────────────────────────────────────

STUDENT_PROFILES = [
    {
        "id": "student_A",
        "description": "struggles with past tense, strong vocabulary",
        "topic": "English grammar - past tense vs present perfect"
    },
    {
        "id": "student_B", 
        "description": "good grammar, weak reading comprehension",
        "topic": "English reading comprehension"
    },
]

def generate_session(student_profile, session_num):
    """Generate a synthetic student-agent interaction session."""
    prompt = f"""Generate a realistic tutoring session between an AI tutor and a student.

Student profile: {student_profile['description']}
Topic: {student_profile['topic']}
Session number: {session_num} of 5

Generate exactly 6 turns (3 student, 3 tutor) as a JSON array:
[
  {{"role": "student", "content": "..."}},
  {{"role": "tutor", "content": "..."}},
  ...
]

Make the student show realistic errors and progress. Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text.strip()
    # Clean up potential markdown fences
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        turns = json.loads(text)
        return turns
    except json.JSONDecodeError:
        # Fallback if parsing fails
        return [
            {"role": "student", "content": f"Session {session_num} student turn 1"},
            {"role": "tutor", "content": f"Session {session_num} tutor response 1"},
            {"role": "student", "content": f"Session {session_num} student turn 2"},
            {"role": "tutor", "content": f"Session {session_num} tutor response 2"},
        ]

def generate_all_sessions(num_sessions=5):
    """Generate sessions for all students."""
    print("Generating synthetic student sessions...")
    all_data = {}
    
    for profile in STUDENT_PROFILES:
        student_id = profile["id"]
        all_data[student_id] = []
        
        for session_num in range(1, num_sessions + 1):
            print(f"  Generating {student_id} session {session_num}...")
            turns = generate_session(profile, session_num)
            all_data[student_id].append({
                "session_num": session_num,
                "turns": turns
            })
            time.sleep(0.5)  # avoid rate limiting
    
    return all_data

# ─────────────────────────────────────────────
# STEP 2: Three memory conditions
# ─────────────────────────────────────────────

def condition_A_append_only(sessions):
    """Condition A: concatenate all session logs."""
    all_turns = []
    for session in sessions:
        for turn in session["turns"]:
            all_turns.append(f"[Session {session['session_num']}] {turn['role'].upper()}: {turn['content']}")
    return "\n".join(all_turns)

def condition_B_rag(sessions, query, top_k=2):
    """Condition B: naive keyword-based retrieval (simulates RAG)."""
    # Simple keyword matching as a proxy for semantic retrieval
    query_words = set(query.lower().split())
    
    scored_sessions = []
    for session in sessions:
        session_text = " ".join([t["content"] for t in session["turns"]])
        session_words = set(session_text.lower().split())
        overlap = len(query_words & session_words)
        scored_sessions.append((overlap, session))
    
    # Return top_k most relevant sessions
    scored_sessions.sort(key=lambda x: x[0], reverse=True)
    top_sessions = [s for _, s in scored_sessions[:top_k]]
    
    retrieved = []
    for session in top_sessions:
        for turn in session["turns"]:
            retrieved.append(f"[Session {session['session_num']}] {turn['role'].upper()}: {turn['content']}")
    return "\n".join(retrieved)

def generate_session_summary(session, student_id):
    """Generate a Tier 2 session summary."""
    turns_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in session["turns"]])
    
    prompt = f"""You are analyzing a tutoring session. Generate a concise structured summary.

Session transcript:
{turns_text}

Return a JSON object with these fields:
{{
  "topics_covered": ["..."],
  "errors_made": ["..."],
  "skills_demonstrated": ["..."],
  "progress_note": "one sentence observation"
}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except:
        return {"topics_covered": [], "errors_made": [], 
                "skills_demonstrated": [], "progress_note": "Summary unavailable"}

def generate_learner_profile(summaries, student_id):
    """Generate a Tier 3 learner profile from session summaries."""
    summaries_text = json.dumps(summaries, indent=2)
    
    prompt = f"""You are building a learner profile from multiple session summaries.

Session summaries:
{summaries_text}

Generate a concise learner profile as JSON:
{{
  "overall_level": "beginner/intermediate/advanced",
  "consistent_strengths": ["..."],
  "persistent_weaknesses": ["..."],
  "learning_trajectory": "improving/plateauing/struggling",
  "recommended_focus": "one sentence recommendation"
}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except:
        return {"overall_level": "unknown", "consistent_strengths": [],
                "persistent_weaknesses": [], "learning_trajectory": "unknown",
                "recommended_focus": "Profile unavailable"}

def condition_C_hierarchical(sessions):
    """Condition C: hierarchical 3-tier compression."""
    print("    Generating session summaries (Tier 2)...")
    summaries = []
    for session in sessions:
        summary = generate_session_summary(session, "student")
        summaries.append({"session_num": session["session_num"], "summary": summary})
        time.sleep(0.3)
    
    print("    Generating learner profile (Tier 3)...")
    profile = generate_learner_profile(summaries, "student")
    
    # Teacher agent receives: profile + last 2 session summaries
    recent_summaries = summaries[-2:]
    
    context = f"LEARNER PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
    context += "RECENT SESSION SUMMARIES:\n"
    for s in recent_summaries:
        context += f"Session {s['session_num']}: {json.dumps(s['summary'], indent=2)}\n"
    
    return context

# ─────────────────────────────────────────────
# STEP 3: Measure tokens and evaluate quality
# ─────────────────────────────────────────────

TEACHER_QUERIES = [
    "What grammar errors has this student made most frequently?",
    "Has this student's performance improved over recent sessions?",
    "What should I focus on in the next class for this student?"
]

def count_tokens_in_string(text):
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4

def evaluate_teacher_response(query, context, ground_truth_context):
    """Use Claude as judge to score teacher agent response quality."""
    
    # Teacher agent answers based on given context
    teacher_response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"You are a teacher assistant. Based on this student information:\n\n{context}\n\nAnswer: {query}"
        }]
    )
    answer = teacher_response.content[0].text
    
    # Judge scores the answer
    judge_response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate this teacher assistant answer on a scale of 1-5.

Question: {query}
Answer given: {answer}
Full student record for reference: {ground_truth_context[:1000]}

Score 1=completely wrong/missing, 3=partially correct, 5=accurate and useful.
Return ONLY a JSON: {{"score": X, "reason": "one sentence"}}"""
        }]
    )
    
    judge_text = judge_response.content[0].text.strip()
    judge_text = judge_text.replace("```json", "").replace("```", "").strip()
    
    try:
        result = json.loads(judge_text)
        return answer, result.get("score", 3), result.get("reason", "")
    except:
        return answer, 3, "Parse error"

# ─────────────────────────────────────────────
# STEP 4: Main experiment runner
# ─────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("EXPERIMENT: Context Engineering for Educational Agents")
    print("=" * 60)
    
    # Generate data
    all_sessions = generate_all_sessions(num_sessions=5)
    
    # Save raw sessions
    with open("sessions_data.json", "w") as f:
        json.dump(all_sessions, f, indent=2)
    print(f"\nGenerated sessions saved to sessions_data.json")
    
    results = {}
    
    for student_id, sessions in all_sessions.items():
        print(f"\n--- Processing {student_id} ---")
        results[student_id] = {}
        
        # Ground truth = full history
        ground_truth = condition_A_append_only(sessions)
        
        # Build contexts for each condition
        print("  Building Condition A (Append-Only)...")
        ctx_A = condition_A_append_only(sessions)
        
        print("  Building Condition B (RAG)...")
        ctx_B = condition_B_rag(sessions, "grammar errors progress performance")
        
        print("  Building Condition C (Hierarchical)...")
        ctx_C = condition_C_hierarchical(sessions)
        
        # Measure tokens
        tokens_A = count_tokens_in_string(ctx_A)
        tokens_B = count_tokens_in_string(ctx_B)
        tokens_C = count_tokens_in_string(ctx_C)
        
        print(f"\n  Token counts:")
        print(f"    A (Append-Only): {tokens_A} tokens")
        print(f"    B (RAG):         {tokens_B} tokens")
        print(f"    C (Hierarchical):{tokens_C} tokens")
        
        # Evaluate quality per query
        print("\n  Evaluating teacher agent responses...")
        query_results = {"A": [], "B": [], "C": []}
        
        for query in TEACHER_QUERIES:
            print(f"    Query: {query[:50]}...")
            
            for cond, ctx in [("A", ctx_A), ("B", ctx_B), ("C", ctx_C)]:
                answer, score, reason = evaluate_teacher_response(
                    query, ctx, ground_truth
                )
                query_results[cond].append(score)
                time.sleep(0.3)
        
        avg_scores = {
            cond: sum(scores)/len(scores) 
            for cond, scores in query_results.items()
        }
        
        print(f"\n  Average quality scores (1-5):")
        print(f"    A (Append-Only): {avg_scores['A']:.2f}")
        print(f"    B (RAG):         {avg_scores['B']:.2f}")
        print(f"    C (Hierarchical):{avg_scores['C']:.2f}")
        
        results[student_id] = {
            "tokens": {"A": tokens_A, "B": tokens_B, "C": tokens_C},
            "quality_scores": avg_scores,
            "query_details": query_results
        }
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    all_tokens_A, all_tokens_B, all_tokens_C = [], [], []
    all_scores_A, all_scores_B, all_scores_C = [], [], []
    
    for student_id, data in results.items():
        all_tokens_A.append(data["tokens"]["A"])
        all_tokens_B.append(data["tokens"]["B"])
        all_tokens_C.append(data["tokens"]["C"])
        all_scores_A.append(data["quality_scores"]["A"])
        all_scores_B.append(data["quality_scores"]["B"])
        all_scores_C.append(data["quality_scores"]["C"])
    
    avg_tok_A = sum(all_tokens_A) / len(all_tokens_A)
    avg_tok_B = sum(all_tokens_B) / len(all_tokens_B)
    avg_tok_C = sum(all_tokens_C) / len(all_tokens_C)
    
    avg_sc_A = sum(all_scores_A) / len(all_scores_A)
    avg_sc_B = sum(all_scores_B) / len(all_scores_B)
    avg_sc_C = sum(all_scores_C) / len(all_scores_C)
    
    print(f"\nAverage tokens per teacher query:")
    print(f"  A (Append-Only):  {avg_tok_A:.0f} tokens  (baseline)")
    print(f"  B (RAG):          {avg_tok_B:.0f} tokens  ({(1-avg_tok_B/avg_tok_A)*100:.1f}% reduction)")
    print(f"  C (Hierarchical): {avg_tok_C:.0f} tokens  ({(1-avg_tok_C/avg_tok_A)*100:.1f}% reduction)")
    
    print(f"\nAverage information quality (1-5 scale):")
    print(f"  A (Append-Only):  {avg_sc_A:.2f}")
    print(f"  B (RAG):          {avg_sc_B:.2f}")
    print(f"  C (Hierarchical): {avg_sc_C:.2f}")
    
    print(f"\nResults saved to results.json")
    print("Done!")
    
    return results

if __name__ == "__main__":
    run_experiment()
