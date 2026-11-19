package ma.emsi.akodad.tp4akodadrag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.emsi.akodad.tp4akodadrag.assistant.Assistant;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 5 : RAG avec récupération sur le Web.
 * Les informations pour le RAG peuvent être récupérées dans le fichier PDF
 * (support de cours sur le RAG) et sur le Web avec le moteur de recherche Tavily.
 * Exercice bonus : un LanguageModelQueryRouter choisit la ou les sources
 * à utiliser, pour que le PDF ne soit pas utilisé quand la question n'a
 * rien à voir avec son contenu (par exemple un événement récent).
 */
public class Test4 {

    public static void main(String[] args) {

        // ===== Configuration du logging =====
        configureLogger();

        // ===== ChatModel (Gemini) =====
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.1-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 du RAG (RAG naïf) : le support de cours sur le RAG =====
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 du RAG =====

        // 1. ContentRetriever pour le fichier PDF (comme le RAG naïf)
        ContentRetriever contentRetrieverPdf = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 2. Moteur de recherche Web Tavily, configuré seulement avec la clé
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(System.getenv("TAVILY_KEY"))
                .build();

        // 3. ContentRetriever qui utilise le moteur de recherche Web
        ContentRetriever contentRetrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        // 4. QueryRouter

        // ===== Version 1 (avant l'exercice bonus) : DefaultQueryRouter =====
        // Les 2 ContentRetrievers sont TOUJOURS utilisés : des extraits du PDF
        // sont ajoutés même pour une question qui n'a rien à voir avec le PDF.
        // QueryRouter queryRouter = new DefaultQueryRouter(contentRetrieverPdf, contentRetrieverWeb);

        // ===== Version 2 (exercice bonus) : LanguageModelQueryRouter =====
        // Le LM choisit la ou les sources à utiliser d'après leur description :
        // pour une question sur un événement récent, il ne choisit que le Web.
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetrieverPdf,
                "Support de cours sur l'intelligence artificielle : fine-tuning, RAG "
                        + "(Retrieval-Augmented Generation), embeddings et modèles de langage (LLM)");
        descriptions.put(contentRetrieverWeb,
                "Recherche sur le Web : actualités, événements récents et informations "
                        + "générales qui ne se trouvent pas dans le support de cours");
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

        // 5. RetrievalAugmentor avec ce QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 6. Assistant avec le RetrievalAugmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

    /**
     * Permet de poser plusieurs questions à l'assistant, saisies au clavier,
     * sans avoir à recompiler. La conversation s'arrête quand on tape "fin".
     * Si l'API du LLM renvoie une erreur, elle est affichée et la conversation continue.
     *
     * @param assistant l'assistant à qui poser les questions
     */
    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (ou \"fin\" pour arrêter) : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                if ("fin".equalsIgnoreCase(question.trim())) {
                    break;
                }
                System.out.println("==================================================");
                try {
                    String reponse = assistant.chat(question);
                    System.out.println("Assistant : " + reponse);
                } catch (RuntimeException e) {
                    System.out.println("Erreur de l'API du LLM : " + e.getMessage());
                    System.out.println("Vous pouvez reposer la question un peu plus tard.");
                }
            }
        }
    }

    /**
     * Configure le logging de LangChain4j (java.util.logging) pour afficher
     * dans la console le détail des requêtes envoyées au LLM et de ses réponses.
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
        packageLogger.setUseParentHandlers(false);
    }
}