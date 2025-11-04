                # Build conversation context summary
                conversation_context = []
                previous_quiz_topics = []
                if len(history) > 1:
                    for turn in history[-3:]:  # Last 3 turns for context
                        q = turn.get('q', '')
                        if q:
                            conversation_context.append(f"Q: {q}")
                
                # Extract topics from previous quizzes in history
                try:
                    for turn in history:
                        a_md = turn.get('a_markdown', '') or turn.get('a', '')
                        # Check if this turn contains a quiz (look for "Quiz:" marker or multiple questions)
                        if 'Quiz:' in a_md or ('Question' in a_md and 'Answer:' in a_md):
                            # Extract question topics from quiz markdown
                            lines = a_md.split('\n')
                            for line in lines:
                                # Match lines like "1. What is...?" or lines with numbered questions
                                if line.strip() and (line[0].isdigit() or line.strip().startswith('-')):
                                    # Extract just the question text (first ~80 chars)
                                    cleaned = line.strip().lstrip('0123456789.-) ').split('\n')[0][:80]
                                    if len(cleaned) > 10 and '?' in cleaned:
                                        previous_quiz_topics.append(cleaned)
                except Exception:
                    pass
                
                # Build comprehensive prompt
                prompt_parts = [
                    f"Generate EXACTLY {question_count} multiple-choice quiz questions based on the following information.",
                    "Focus on the main answer content, but also consider the conversation context and document sources.",
                ]
                
                if previous_quiz_topics:
                    prompt_parts.extend([
                        "",
                        "IMPORTANT: The user has already been quizzed on these topics. Generate NEW questions covering DIFFERENT aspects:",
                        "\n".join([f"- {t}" for t in previous_quiz_topics[-20:]]),  # Last 20 quiz questions
                        "",
                    ])
                
                prompt_parts.extend([
                    f"Return ONLY a JSON array of length EXACTLY {question_count} where each question is an object with:",
                    "- question (string): the quiz question",
                    "- options (array of 4 strings): the answer choices",
                    "- correct (integer 0-3): index of the correct answer",
                    "- explanation (string): why the correct answer is right",
                    "",
                    "IMPORTANT: Distribute correct answers randomly across all four positions (A/B/C/D). Do NOT favor any position.",
                    "Aim for roughly equal distribution: ~25% each for indices 0, 1, 2, and 3.",
                    "",
                    "PREVIOUS ANSWER (main content to quiz on):",
                    previous_answer[:2000],  # Limit length
                    ""
                ])
