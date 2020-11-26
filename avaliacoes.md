---
layout: page
title: Avaliações
description: Descrição das Avaliações
---

# Projeto Final

{: .no_toc .mb-2 }

Create something you want to share.[^1]
{: .fs-6 .fw-300 }

[^1]: CS50. 2020. Final Project. In CS50 Spring 2020. <https://cs50.harvard.edu/college/2020/spring/project/>

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

The final project is your opportunity to create something new that you want to share with the rest of the world. The nature of your project is entirely up to you so long as it is sufficiently complex. All that we ask is that you build something of interest to you, that you solve an actual problem, that you impact campus, or that you change the world. Strive to create something that outlives this course.

**The final project must involve developing an algorithm that either responds to user input or answers a question given a dataset. The problem should be at least as complex as the individual take-home assessments.** The project should be primarily implemented in Java (recommended), [Dart](https://dart.dev/), [Python](https://www.python.org/), [Go](https://golang.org/), or any flavor of [JavaScript](https://www.javascript.com/). (Other languages are possible with additional approval.) You are welcome to utilize any infrastructure provided the staff can access any required software and hardware. That said, the staff is best able to support projects that run in an Ed Workspace and use the class's standard web app framework.

The final project is a collaborative activity: everyone is required to work in groups of either 2 or 3 students. You may not work alone. It is expected that every student contributes equally to the design and implementation of that group's project. We expect each student to contribute about 18--24 hours of total labor towards the project since it represents 3 weeks of work in the course and replaces several class sessions. Although no more than three students may design and implement a given project, you are welcome to solicit advice from others so long as the final implementation represents your group's own original work.

## Ideation

The ideation phase is your opportunity to bounce one or more ideas off of the staff and identify collaborators. There's no formal submission required during ideation.

- What idea(s) do you have for your project?
- If you plan to collaborate with one or two other students, what are their names?
- When can all of the collaborators meet?

You might have multiple ideas in mind. TAs and potential collaborators can help you choose between different options.

> We want to build a [pattern-matching chatbot](https://en.wikipedia.org/wiki/ELIZA) web app.

## Proposal

The proposal is your opportunity to receive feedback and approval from the staff before you proceed to design. The staff will either approve your proposal or require modifications on your part for subsequent approval. Your proposal, even if approved, is not binding; you may alter your plan at any point, provided you obtain the staff's approval for any modifications.

- What is the working title for your project?
- Pitch your project in one or two sentences.
- Describe your project in one or more paragraphs. What will your software do? What features will it have? How will it be executed?

> FRANK: A frank chatbot. FRANK is a web app that accepts text input from the user, parses it to identify keywords and important phrases, and returns a coherent sentence that mentions the keywords and phrases in a natural way.

In the world of software, most everything takes longer to implement than you expect. It's not uncommon to accomplish less in a fixed amount of time than initially hoped.

- Define a **good** outcome for your project. What functionality will you definitely accomplish?
- Define a **better** outcome for your project. What do you think you can accomplish by the deadline?
- Define a **best** outcome for your project. What do you hope to accomplish by the deadline?

> A good outcome would be a chatbot that forms coherent sentences by mentioning keywords and important phrases from the input text. A better outcome would be a chatbot that also models conversation history, including an option to start a new conversation by clearing the history. A best outcome would be a chatbot that can learn from successful past conversations: the chatbot stores the conversation history of each user that had a positive experience so that future conversations with new users can draw on keywords and themes that were helpful to other users.

Finally, outline your next steps in one or more paragraphs.

- What new skills will you need to acquire?
- What topics will you need to research?
- If working with one or two other students, who will do what?

> A pattern-matching chatbot will require more investigation into language modeling and syntax. There might be Java libraries on the internet that can tag parts of speech in a sentence to help the app identify keywords and important phrases. We can also draw inspiration from other programmers that have written about their experience creating chatbots.

If you choose to do research, it's important to consider how other programmers conceptualize the problem. In the chatbot example, our good outcome could be solved using string manipulations and dictionary lookups to identify parts of speech. If we model our project after examples that conceptualize the problem in a more complex way, it can be easy to get sidetracked and end up working on an approach that doesn't address the original proposal!

## Status report

The status report is your opportunity to keep yourself on track by meeting with a TA to discuss your project progress and adapt plans as necessary. The status report meeting will be most helpful if the project group has already made some progress into the project implementation.

- What have you done for your project so far?
- What have you not done for your project yet?
- What problems, if any, have you encountered?

## Implementation

Create an Ed workspace to collaborate during the implementation of your project. Include any and all files required to run your software. Keep in mind our code quality guidelines when implementing your project. Source code should meet our code quality guidelines so that other programmers can understand it. If you incorporate code found online, cite the source with inline comments. In your selection of code sources, [be reasonable](https://apps.leg.wa.gov/WAC/default.aspx?cite=478-121-107). This project should be primarily original work.

To run Java code in a workspace, you will need to use the terminal. For example, if we had a `ChatBot` class that includes a `main` method, we could run the following terminal command. (The `rm` command at the end removes the `.class` files generated by Java after the program is done running.)

```sh
javac ChatBot.java && java ChatBot; rm *.class
```

If the `ChatBot` requires other Java libraries (`jar` files), include the `-cp` flag to tell Java where to find `YOUR_JAR_FILES`.

```sh
javac -cp ".:YOUR_JAR_FILES" ChatBot.java && java -cp ".:YOUR_JAR_FILES" ChatBot; rm *.class
```

In addition to the source code for running your project, include three additional text files.

### User guide

Write a user guide for your project in the form of a file called `README.txt` or `README.md` (if you prefer Markdown) at least several paragraphs in length. Though the structure of your user guide is entirely up to you, it should be clear to the staff how and where, if applicable, to compile, configure, and use your project. It should not be necessary for us to contact you with questions regarding your project after its submission. Hold our hand with this documentation: be sure to answer in your documentation any questions that you think we might have while testing your work.

The guide should also include a link to a **short video** (no more than 5 minutes long) that presents your project to the world with slides, screenshots, voiceover, and/or live action. Your video should somehow include your project title, your names, and any other details that you'd like to convey to viewers. We recommend recording a video through Zoom for ease of screensharing and so that all collaborators can present.

### Design document

Include your project's design document in the form of a file called `DESIGN.txt` or `DESIGN.md` at least several paragraphs in length. The design document discusses the project's technical implementation details and reasoning behind design decisions.

### Reflection

In a file named `reflection.txt`, write a reflection answering the following questions (along with anything else you find appropriate) at least one paragraph in length.

- What did you learn during this project?
- What did you enjoy during this project?
- What did you find challenging or frustrating during this project?
- What did you find particularly helpful for your learning during this project?
