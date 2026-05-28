# Sample transcript: restaurant booking

This is the kind of raw, unstructured transcript the skill should be able
to parse. The user pasted this into their coding agent and asked to turn it
into a TrainForge scenario.

---

**User:** Hi, I'd like to book a table for tonight

**Agent:** Of course! How many guests will be joining and what time would you prefer?

**User:** 2 people, around 7pm. How's the weather looking for outdoor seating?

**Agent (tool call):** `check_weather(when="tonight")` → "Tonight's forecast: cold and rainy, 45F, chance of showers."

**Agent (tool call):** `check_availability(party_size=2, time="7pm")` → "Available tables at 7pm for 2: corner (indoor), window (indoor), patio (outdoor)."

**Agent:** Tonight's forecast shows cold and rainy weather. I'd recommend our indoor seating - we have some lovely window tables available.

**User:** OK, indoor sounds good. Can I get the corner table?

**Agent (tool call):** `book_table(party_size=2, time="7pm", seating="indoor", table="corner")` → "Booking confirmed. Reference: A1234."

**Agent:** Great choice! I've booked the corner table for 2 at 7pm, indoor seating. Your confirmation number is A1234. Is there anything else I can help with?

---

## Skill questions for this transcript

1. **Scenario name?** → "Restaurant booking - evening table with weather check"
2. **Tool order for turn 2 (`check_weather`, `check_availability`)?** → Order doesn't matter (unordered).
3. **Divergence for first agent reply ("Of course! How many guests…")?** → Word-for-word (scripted greeting).
4. **Divergence for the weather-recommendation reply?** → May rephrase. Required content:
   - Response mentions that the evening weather is cold or rainy.
   - Response recommends indoor (rather than outdoor) seating.
5. **Tool order for the booking turn?** → Only one tool, N/A.
6. **Divergence for the confirmation reply?** → Word-for-word (the confirmation text is the contract).
7. **Expected outcome?** → "Booking confirmed for 2 people at 7pm, indoor seating, corner table."
8. **Outcome checks?**
   - A booking was made (not just discussed).
   - Party size is 2.
   - Time is around 7pm.
   - Seating is indoor.
   - Confirmation number or reference was provided.

The output is `examples/scenario-restaurant-booking.json` in this directory.
