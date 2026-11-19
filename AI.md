# Utilisation de l'IA – TP 4 : RAG avec LangChain4j

## Vue d'ensemble

Pour ce TP, j'ai utilisé Claude comme assistant afin de m'aider à comprendre et résoudre certaines difficultés rencontrées pendant le développement de mon application RAG avec LangChain4j.

J'ai exécuté moi-même le code dans IntelliJ, effectué les tests et analysé les résultats avec l'aide de l'IA. Lorsque je rencontrais un problème, je transmettais les messages d'erreur ou les logs de la console pour obtenir des explications et identifier les corrections nécessaires.

Voici les situations dans lesquelles l'IA m'a été particulièrement utile.

---

## 1. Le programme restait bloqué lors de l'appel à Gemini

**Problème rencontré :** au premier lancement de `RagNaif`, le programme restait bloqué pendant plus de 10 minutes sans afficher de réponse ni de message d'erreur.

**Pourquoi j'ai utilisé l'IA :** comme aucune erreur n'apparaissait, je ne savais pas si le problème venait du chargement des documents, de la génération des embeddings ou de l'appel à Gemini.

**Prompt utilisé :** j'ai transmis la sortie de la console à l'IA, puis les résultats des différents tests qu'elle m'a proposés.

**Réponse de l'IA :** l'IA m'a conseillé d'ajouter des `System.out.println` entre les différentes étapes pour identifier le blocage. Nous avons constaté que les 355 segments et leurs embeddings étaient correctement générés, mais que l'appel à Gemini ne se terminait pas.

Elle m'a ensuite proposé de tester directement l'API avec `curl` et d'utiliser `jstack` pour examiner l'état du programme. Les tests suivants ont révélé des erreurs `503 UNAVAILABLE` et `429 RESOURCE_EXHAUSTED`, liées à la surcharge du modèle et aux limites de quota.

**Ce que j'ai appris :** j'ai appris à localiser un blocage en isolant les différentes étapes d'un programme et à utiliser `jstack` pour examiner les threads Java. J'ai également compris que certaines bibliothèques réessaient automatiquement les requêtes en cas d'erreur, ce qui peut donner l'impression que le programme est bloqué lorsqu'aucun logging n'est configuré.

---

## 2. Comprendre pourquoi le logging ne fonctionnait pas

**Problème rencontré :** après avoir ajouté la méthode `configureLogger()` de l'énoncé, aucun log LangChain4j ne s'affichait. La console indiquait `SLF4J: No SLF4J providers were found`. Après une première correction, les logs apparaissaient en double.

**Pourquoi j'ai utilisé l'IA :** je ne comprenais pas pourquoi la configuration du logger Java ne suffisait pas à afficher les requêtes envoyées au LLM.

**Prompt utilisé :** j'ai demandé à l'IA de m'aider à modifier le code pour activer le logging, puis je lui ai envoyé la sortie de la console avec les logs dupliqués.

**Réponse de l'IA :** l'IA m'a expliqué que LangChain4j utilise SLF4J, tandis que la méthode de l'énoncé configurait `java.util.logging`. Elle m'a conseillé d'ajouter la dépendance `slf4j-jdk14` et d'activer `.logRequestsAndResponses(true)`.

Pour résoudre le problème des logs en double, elle m'a proposé d'utiliser `setUseParentHandlers(false)`.

**Ce que j'ai appris :** j'ai compris la différence entre SLF4J et `java.util.logging`, ainsi que le rôle des handlers dans la propagation des logs. J'ai surtout compris l'importance du logging pour une application RAG, car il permet de voir les documents récupérés et le contexte réellement transmis au LLM.

---

## 3. Comprendre pourquoi la deuxième question donnait une mauvaise réponse

**Problème rencontré :** après avoir demandé « Qu'est-ce que le fine-tuning ? », j'ai posé la question « Dans le support de cours fourni, à quoi c'est comparé ? ». Le LLM donnait une réponse incohérente alors que l'information était présente dans le document.

**Pourquoi j'ai utilisé l'IA :** je voulais comprendre pourquoi le RAG ne retrouvait pas le bon passage du cours malgré la mémoire de conversation.

**Prompt utilisé :** j'ai transmis à l'IA les logs complets des deux questions pour qu'elle analyse les segments récupérés.

**Réponse de l'IA :** l'IA m'a montré que le retriever recherchait uniquement à partir de la deuxième question, sans comprendre que « c' » faisait référence au fine-tuning.

Elle m'a expliqué comment utiliser `CompressingQueryTransformer` pour reformuler la question à partir de l'historique de conversation, puis intégrer ce mécanisme avec un `RetrievalAugmentor`.

**Ce que j'ai appris :** j'ai compris que la mémoire de conversation du LLM ne garantit pas que le retriever utilise cet historique. Le QueryTransformer permet de reformuler une question dépendante du contexte avant d'effectuer la recherche documentaire. J'ai également compris que cette transformation nécessite un appel supplémentaire au LLM.

---

## 4. Comprendre pourquoi le RAG utilisait des documents sans rapport avec la question

**Problème rencontré :** lors du test avec plusieurs sources (PDF et Web), j'ai demandé « Qui a gagné la Coupe du monde de football 2026 ? ». Les logs montraient que des passages du PDF sur le RAG étaient ajoutés à la requête, alors qu'ils n'avaient aucun rapport avec le football.

**Pourquoi j'ai utilisé l'IA :** je voulais comprendre pourquoi le système récupérait des documents non pertinents et comment éviter d'envoyer des informations inutiles au LLM.

**Prompt utilisé :** j'ai transmis les logs de la requête à l'IA pour analyser les documents sélectionnés.

**Réponse de l'IA :** l'IA m'a expliqué que le `DefaultQueryRouter` envoyait la question à tous les retrievers. Le retriever vectoriel pouvait donc sélectionner les segments les plus similaires du PDF, même lorsque leur pertinence était faible.

Elle m'a proposé d'utiliser un `LanguageModelQueryRouter` avec une description de chaque source afin que le LLM choisisse entre le document de cours et la recherche Web.

Après cette modification, les logs ont montré que la question sur la Coupe du monde utilisait uniquement le Web, tandis que la question sur le RAG utilisait le PDF.

**Ce que j'ai appris :** j'ai compris que la recherche vectorielle ne garantit pas toujours la pertinence des documents récupérés. J'ai aussi appris qu'un routage intelligent permet de sélectionner les sources adaptées à chaque question, de limiter les informations inutiles et de réduire le nombre de tokens envoyés au modèle.

---

## Bilan

L'IA m'a principalement aidé à diagnostiquer des problèmes difficiles à identifier et à mieux comprendre le fonctionnement interne d'une application RAG.

Les explications sur le logging, la transformation des questions et le routage des sources m'ont permis de comprendre comment les différents composants de LangChain4j interagissent.

J'ai appliqué les corrections dans mon projet et vérifié leur fonctionnement à partir des résultats d'exécution et des logs de la console.