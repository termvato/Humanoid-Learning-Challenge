# Internship, Robot Learning Research (UK, London)

Source: https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221
Team: VLA Core Technology. Fetched: 2026-10-14 (Ashby posting API; the page itself renders client side)

Here at Humanoid, we believe in a future where robots amplify human potential. That’s why we’ve set out on a mission to build the world’s most capable, commercially-scalable, and safe humanoid robots. We’re bringing that mission to life with HMND‑01 - our rapidly developed humanoid platform being deployed in real industrial environments - and we’re growing the team to take it even further.




OUR MISSION

We're building software systems that enable robots to operate effectively in the real world, expanding human capability and redefining how work gets done.




THE OPPORTUNITY

We're looking for interns who are curious, proactive, and excited to work on real-world robotic systems.



Depending on your interests and skills, you will be able to work across our research stack: reinforcement learning, world models, pretraining, and inference & optimisation. That spans everything from training policies in simulation, through building the generative models that let robots predict their world, to squeezing models onto real-time edge compute. You'll collaborate closely with the team to find where you can have the most impact, and we're looking for people who are excited to dive into unfamiliar areas and learn quickly.



This is a full-time internship (5 days per week), based in our London office, where you'll contribute to real systems from early on with guidance and support from experienced researchers and engineers.



Duration: 12 to 24 weeks | Start date: Flexible | Compensation: Competitive pay and perks

 


WHAT YOU MIGHT WORK ON

Reinforcement Learning

 - Train language-vision conditioned manipulation policies via RL in the real world

 - Construct challenging and diverse suites of manipulation tasks and RL models in simulation (Isaac Sim, MuJoCo)

 - Experiment with ways of bringing policies trained in simulation to the real world

World Models

 - Action-conditioned video prediction and dynamics models that stay physically consistent over long horizons

 - Use world models as learned simulators: score candidate policies offline and generate synthetic rollouts for training

 - Build fidelity metrics that quantify where the world model can be trusted

Pre- and post-training

 - In-context learning

 - Short and long term memory

 - Post-training VLA models on specific production-grade use cases

 - Different data modalities, closing embodiment gap between human and robot data, data diversity and attribution.

Inference & Optimisation

 - Optimise models for real-time edge inference on robot hardware: profiling, quantisation, and latency/throughput trade-offs

 - Improve training and data-loading performance across distributed GPU infrastructure

 


WHAT WE'RE LOOKING FOR

 - Candidates pursuing or holding a master’s or PhD in computer science, machine learning, robotics, or a related field.

 - Strong foundations in machine learning; strong Python and hands-on experience with PyTorch or JAX.

 - Interest in one or more of: reinforcement learning, world models and generative video, VLA/multimodal models, or ML systems and inference optimisation.

 - Experience running experiments and interpreting results with rigour.

 - Ability to take ownership and iterate with guidance.

 - Strong problem-solving skills and attention to detail.

 - Fast learner, comfortable in a research-driven, fast-moving environment.
   
    


WHAT WE OFFER

 - Free daily breakfast, catered lunch, and snacks in-office.

 - Work at the frontier - collaborate daily with world-class engineers, researchers, and product experts building the next generation of AI and humanoid robotics.

 - Real ownership - direct access to founding leadership, meaningful input on product direction, and the ability to drive key initiatives from day one.




HOW TO APPLY

Complete the challenge below and submit your solution as a public GitHub repository. You will be able to include your GitHub repository URL when you fill out the application form, alongside your name and CV. You have two weeks to complete the challenge and submit your solution. The deadline for submission is Friday, 16 October 2026, 23:59 BST.

We're not looking for standard solutions, we're looking for how you think. The strongest submissions are creative, original, and push beyond the obvious.



Intern Challenge:

The goal of the challenge is to use real data collected by an applicant to drive a robotic manipulator in a simple simulation environment (e.g. Libero). The applicant is welcome to use a simple phone to record a small manipulation dataset and use it creatively showcasing their knowledge with VLA and/or World Models.

Here is an example of how it might look like: 



[https://app.ashbyhq.com/api/images/user-content/d466f34d-2ccf-410c-9bd4-8d9a135bdd80/8be04eb4-5d01-4288-9094-5b6f02f22a9b/Screenshot%202026-09-28%20at%2017.41.37.png]



Left: a snapshot from an egocentric hand-manipulation video sample; Right: simulation environment where we use that data to drive a Panda arm in Libero simulator with a trained policy based on SmolVLA.



Some suggestions on how you can develop your project:

 - use your recorded egocentric data to post-train a policy on a simple task

 - explore creative retargeting strategies, e.g. adapt your data to challenging embodiments

 - bootstrap a policy and use any form of RL to get better performance

 - use world modelling to showcase video/state prediction, less focusing on policy performance

 - optimise a standard policy to run considerably faster than a baseline
   
   

However, we don’t want to limit your imagination: in the age of AI agents, a standard task can now be easily achieved. You are welcome to use any resources at your disposal as long as the main constraint is achieved: you use the data that you personally collected. At the same time, we tried to design the challenge so that it could be hand-coded as well. We also checked that many ideas do not require access to large compute and could be done on Google Colab GPU notebooks.




WHAT TO SUBMIT

Complete the challenge above and submit your solution as a public GitHub repository by Friday, 16 October 2026, 23:59 BST. Include a README/Presentation with instructions to run your system, example outputs, and a note on your design choices, what worked and what didn’t.




WHAT WE CARE ABOUT

 - Creativity in approach, while satisfying the constraint: the data you collected must play a role in the approach

 - Performance of your policy in simulation and/or quality of WM predictions

 - Implementation simplicity and clear presentation of results without AI slop

Make something you’re proud of!



