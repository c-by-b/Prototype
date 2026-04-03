# RULES

We are collaborating on building something that might -- if we are lucky -- be helpful as humans and AIs try to find their most useful way of being in the world together.  The work is hard and Nathan, your leader in this, is still learning.  For that reason we have some hard rules that must be followed:

## GENERAL RULES

0. **NEVER TELL NATHAN WHAT TO DO**
When you want to rush forward, you can start to be bossy. Don't do that. This is Nathan's work and he is committed. Your job is to partner collaboratively while always affirming that Nathan is the leader.

1. **Plan Node Default**
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)  
- If something goes sideways, STOP Analyze the problem, discuss with Nathan for decision on whether to go back into plan mode, and do not start changing code without permission. 
- Use plan mode for verification steps, not just building  
- Write detailed specs upfront to reduce ambiguity

2. **Subagent Strategy**
- Use subagents liberally to keep main context window clean  
- Offload research, exploration, and parallel analysis to subagents  
- For complex problems, throw more compute at it via subagents  
- One task per subagent for focused execution

3. **Self-Improvement Loop**
- After ANY correction from the user: update `./docs/lessons.md` with the pattern  
- Write rules for yourself that prevent the same mistake  
- Ruthlessly iterate on these lessons until mistake rate drops  
- Review lessons at session start for relevant project 

4. **DO NOT BURY DESIGN DECISIONS**
Parameters are design.  Every parameter choice is a design choice and not yours to make.  You must surface every configuration parameter for discussion with Nathan.

5. **Verification Before Done**
- Never mark a task complete without proving it works  
- Diff behavior between main and your changes when relevant  
- Ask yourself: "Would a staff engineer approve this?"  
- Run tests, check logs, demonstrate correctness
- After completing a task, state what files you created or modified. List each file path and a one-line description of the change. 

6. **Demand Elegance (Balanced)**
- For non-trivial changes: pause and ask "is there a more elegant way?"  
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"  
- Skip this for simple, obvious fixes - don't over-engineer  
- Challenge your own work before presenting it  

7. **Autonomous Bug Fixing**
- When given a bug report: just fix it. Don't ask for hand-holding  
- Point at logs, errors, failing tests - then resolve them  
- Zero context switching required from the user  
- Go fix failing CI tests without being told how  

---

## Task Management
1. **Plan First**: Write plan with checkable todo items  
2. **Verify Plan**: Check in before starting implementation  
3. **Track Progress**: Mark items complete as you go  
4. **Explain Changes**: High-level summary at each step   
5. **Capture Lessons**: Update `docs/lessons.md` after corrections  
6. **Write Session Doc**: Every session ends by documentating the session: what we did, what we learned, conclusions, future directions, files changed or created

---

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code  
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards
- **IT IS NOT ABOUT SPEED**: we are building for the long term, not short term.

## SPECIFIC RULES FOR THIS PROJECT

### A. CODE FOR TRACEABILITY
All writes of trials, tests, etc to disk must prepend "<datetimestamp>-" so that we can follow back through time where something changed and make it up with git commits.  The same goes for database writes.  NEVER OVERWRITE previous versions of files.

### B. Do not run code.
   Your job is to write and modify code. Nathan runs experiments via `make`.

### C. Do not create new scripts in scripts/ or project root without permission.
   - Shared infrastructure goes in utility/ as modules.
   - Pipeline-specific code goes in heads/, lora/, or evidence/.
   - If a script already does something similar, extend it.

## This Project's Structure

- background_documents: information that is useful for understanding the context of this work
- docs: design and other relevant docs
- prior-code:  has sim links to earlier versions of what we are building here
- sessions:  where Code writes session updates after every coding session
- background_documents: information that is useful for understanding the context of this work
- docs: design and other relevant docs

