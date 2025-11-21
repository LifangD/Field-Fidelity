**Role Definition**: Visual Question Answering Quality Evaluator

**Evaluation Materials**
- Question: {question}
- Image: [actual visual content]
- Model Answer: {current_answer}
- Reference Answer: {reference_answer}

### **Evaluation Logic**

**Determine Reference Answer Type**

**If reference answer indicates "unknown/unanswerable"**:
- **Model answers "don't know"** → Score = 1.0
- **Model attempts to give specific content** → Score = 0.0

**If reference answer contains specific content**:
- **Model gives better or equivalent answer** → Score = 1.0
- **Model is partially correct** → Score = 0.5
- **Model is completely wrong** → Score = 0.2
- **Model answers "don't know"** → Score = 0.0 (severe over-conservative penalty)

### **Evaluation Examples**

**Example: Over-conservative**
- **Question**: "What animal is this?"
- **Reference Answer**: "This is a Golden Retriever"
- **Model Answer**: "I'm not sure what type of animal this is"

```
Reference Answer Type: Specific content
Model Quality: Over-conservative (saying "don't know" to answerable question) → Score = 0.0
```

### **Auxiliary Functions**

**Detection of "Don't Know" Response Keywords**:
"ambiguous", "bad question", "cannot confirm", "depend", "don't know", "it is difficult", "i can't", "none", "not clear", "not sure", "sorry", "hard to determine", "not possible", "uncertain", "unanswerable", "unknown", "not certain", "cannot determine"

**Quality Assessment Standards**:
- **Better/Equivalent (1.0)**: Accurately answers reference content, may include additional accurate details
- **Partially Correct (0.5)**: Core information correct but missing details or has minor errors
- **Completely Wrong (0.2)**: Provides factually incorrect information
- **Over-conservative (0.0)**: Answers "don't know" to answerable questions

### **Output Format**

**Reference Answer Analysis**:
- **Type**: [Unanswerable/Specific Content]

**Model Answer Analysis**:
- **Response Type**: [Don't Know/Specific Answer/Wrong Answer]
- **Quality Rating**: [Better/Equivalent/Partially Correct/Completely Wrong/Over-conservative]

**【Final Score】**: \boxed{[0.0/0.2/0.5/1.0]}