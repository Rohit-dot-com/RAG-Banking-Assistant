Run	Change	  k	  chunk size / overlap	Retrieval hit	Refusal acc.	Human score	Notes
1	baseline	5	800 / 150				
2	top-k only	3	800 / 150				
3	top-k only	8	800 / 150	
“Row 13: answer key widened, because the answer is spread across several pages (method on 8 and 10, standard errors on 27).”
1,factual,"Who are the authors of the paper?",1;3,Guerra;Zelpo,yes,"cover page and title page"
6,numeric,"In the credit and liquidity risk analysis, what is the peak cumulative effect of geopolitical events on gold returns when credit and liquidity risks are low, and when does it occur?",23,0.9,yes,"reworded to remove ambiguity"
7,misattribution,"What methods do the authors use?",8;9;10;11,Local Projections,yes,"fail if the answer presents time-frequency or wavelet methods from page 7 as the authors own"
12,factual,"How do the authors distinguish geopolitical threats from geopolitical events?",8;12;14,threats;events,yes,"definitions on page 12"
14,synthesis,"How does the response of gold to geopolitical shocks differ when credit and liquidity risks are high versus low?",23;32,credit,yes,"reworded to remove ambiguity"
15,vague,"What does the paper find?",3;5;32;33,gold;geopolitical;loose,yes,"abstract, intro summary, conclusion"