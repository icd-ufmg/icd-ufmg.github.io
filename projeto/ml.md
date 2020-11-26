---
layout: default
title: Aprendizado de Máquina
parent: Projeto
nav_order: 4
---

# Zen
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Boolean zen

Boolean zen is all about using boolean values efficiently and concisely.

The following code shows a tempting way to write an if statement based on boolean value.

Bad
: ```java
  if (test == true) {
      // do some work
  }
  ```

Note that `test` itself is a `boolean` value. When it is `true`, we are asking whether `true == true`. `true == true` will evaluate to `true`, but remember that the `if` branch will execute as long as what is in its parentheses evaluates to true. So we can actually use `test` directly:

Good
: ```java
  if (test) {
      // do some work
  }
  ```

Note
: To check for the opposite of `test`, don't check for `test == false`. Instead, use `!test` to check that the opposite of `test` evaluates to `true`.

Here's an example that uses what we learned in the previous section about simplifying boolean expressions.

Bad
: ```java
  if (test) {
      return true;
  } else {
      return false;
  }
  ```

There is actually a much more concise way to do this. If we want to return true when test is true and false when test is false, then we actually just want to return the value of `test`.

Good
: ```java
  return test;
  ```

In general, make your use of booleans to simplify conditional and boolean expressions. This is something to look out for whenever you have conditions or boolean values.

Bad
: Because this code always wants to do one or the other, (and doesn't involve a return or exception) we want to express this code more simply as an if/else.

  ```java
  if (someTest) {
      System.out.println("hello!");
  }
  if (!someTest) {
      System.out.println("I'm redundant");
  }
  ```

Bad
: Note that the behavior inside this if block is exactly the same behavior as in the other if block. Instead of rewriting the same code twice, we can combine the two if conditions with `||` and just write the behavior once.

  ```java
  if (max < result) {
      return max;
  }
  if (max == 0) {
      return max;
  }
  ```

Bad
: It doesn't matter if you think of conditions/cases or their negated versions, but after revising your code don't include empty condition blocks with no line of code inside. Instead, just flip the condition to have if (max >= 0), and no else.

  ```java
  if (max < 0) {
      // do nothing
  } else {
      ...
  }
  ```

## Loop Zen

### Loop Bounds

When writing loops, choose loop bounds or loop conditions that help generalize code the best. For example, the code before this `for` loop is unnecessary.

Bad
: ```java
  System.out.print("\*");
  for (int i = 0; i < 4; i++) {
      System.out.print("\*");
  }
  ```

Instead, why not just run the loop one extra time? This way we avoid writing duplicate behavior.

Good
: ```java
  for (int i = 0; i < 5; i++) {
      System.out.print("\*");
  }
  ```

### Only Repeated Tasks

If you have something that only happens once (maybe at the end of a bunch of repeated tasks), then don't put code for it inside of your loop.

Bad
: ```java
  for (int i = 0; i < 4; i++) {
      if (i != 3) {
          System.out.println("working hard!");
      } else {
          System.out.println("hardly working!");
      }
  }
  ```

Good
: ```java
  for (int i = 0; i < 3; i++) {
      System.out.println("working hard!");
  }
  System.out.println("hardly working!");
  ```

Similarly, a loop that always runs one time is also a bad use of a loop. When reviewing the logic of your code, it might be helpful to check if a loop can be reduced to an if or just deleting the loop entirely.

## Recursion zen

**Avoid adding extra if/else cases to recursive methods when there are more generalized conditions to use.**

Bad
: This private method is an example of **arm's length recursion**: it has an extra base case for `index == list.length - 1`. If we omit this and just use the first base case, the behavior will actually still generalize and be correct, but our code will be more concise and easier to reason about.

  ```java
  private static int sum(int[] list, int index) {
      if (index == list.length) {
          return 0;
      } else if (index == list.length - 1) {
          return list[index];
      } else {
          return list[index] + sum(list, index + 1);
      }
  }
  ```

Good
: ```java
  private static int sum(int[] list, int index) {
      if (index == list.length) {
          return 0;
      } else {
          return list[index] + sum(list, index + 1);
      }
  }
  ```

Bad
: This example also exhibits arm's length recursion since the code in the public method would be more appropriate to handle in the private helper method.

  ```java
  public static int sum(int[] list) {
      if (list.length < 1) {
          return 0;
      } else {
          return list[0] + sum(list, 1);
      }
  }

  private static int sum(int[] list, int index) {
      if (index == list.length) {
          return 0;
      } else {
          return list[index] + sum(list, index + 1);
      }
  }
  ```

Good
: ```java
  public static int sum(int[] list) {
      return sum(list, 0);
  }

  private static int sum(int[] list, int index) {
      if (index == list.length) {
          return 0;
      } else {
          return list[index] + sum(list, index + 1);
      }
  }
  ```
