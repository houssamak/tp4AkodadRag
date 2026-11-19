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
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.akodad.tp4akodadrag.assistant.Assistant;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 4 : pas de RAG.
 * Un QueryRouter personnalisé demande au LM si la question porte sur l'IA.
 * Si le LM répond "non", aucun ContentRetriever n'est utilisé (pas de RAG) ;
 * sinon, le RAG est fait avec le support de cours sur le RAG.
 */
public class Test3 {

    public static void main(String[] args) {

        // ===== Configuration du logging =====
        configureLogger();

        // ===== ChatModel (Gemini) =====
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash")
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 du RAG : seulement le support de cours sur le RAG =====
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 du RAG =====
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Template de prompt avec un paramètre pour la requête de l'utilisateur
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{requete}}' porte sur l'IA ? "
                        + "Réponds seulement par 'oui', 'non' ou 'peut-être'.");

        // QueryRouter personnalisé : classe anonyme interne à la méthode main.
        // Le LM décide s'il faut faire du RAG ou pas.
        QueryRouter queryRouter = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                // Le paramètre {{requete}} est remplacé par le texte de la question
                Prompt prompt = promptTemplate.apply(Map.of("requete", query.text()));
                // Question posée directement au ChatModel (sans assistant AiServices)
                String reponse = model.chat(prompt.text());
                if (reponse.trim().toLowerCase().startsWith("non")) {
                    // Pas de RAG : aucun ContentRetriever
                    return Collections.emptyList();
                }
                // "oui" ou "peut-être" : RAG avec le support de cours
                return List.of(contentRetriever);
            }
        };

        // RetrievalAugmentor avec ce QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Assistant
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