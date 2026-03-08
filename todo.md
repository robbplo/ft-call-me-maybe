 Missing entirely:                                                             
  1. README.md (section VI) — not present at project root. Must include: project
   description, instructions, algorithm explanation, design decisions,          
  performance analysis, challenges, testing strategy, and example usage.        
                                                                                
  Code quality (IV.1):                                                          
  2. Docstrings — no PEP 257 docstrings on any class or function.               
  3. Type hints — incomplete in several places (e.g. function_call_generator.py,
   function_selector.py missing return types on some methods).                  
                                                                                
  Potential violation:                                                          
  4. import torch in src/constrained_json_decoder.py:6 — subject IV.3.1         
  explicitly forbids pytorch. It's only used for the torch.Tensor type          
  annotation in the Tokenizer protocol. Could be replaced with a plain type or  
  typing.Any.                                                                   
                                                                                
  Error handling gaps:                                                          
  5. FunctionSelector._load_functions — no try/except around file read; a       
  missing or malformed functions_definition.json will crash with an unhandled   
  exception.                                                                    
  6. FunctionSelector.select_function — if the LLM doesn't converge to a valid  
  function name, self.functions[result] will raise KeyError.                    
                                                                                
  Minor:                                                                        
  7. Input format — subject's example shows a plain string array ["prompt1",    
  ...], but your code expects [{"prompt": "..."}]. Your actual input file       
  matches what the code expects, but worth noting in case the evaluator's test  
  file differs.                                                                 
  8. data/output/ in submission — subject says don't include it. The .gitignore 
  in that folder likely handles it, but worth confirming it excludes the        
  generated JSON.                                                               
                                                                                
  The biggest actionable gaps are README.md, docstrings, and the torch import.  
  Want me to tackle any of these?        
