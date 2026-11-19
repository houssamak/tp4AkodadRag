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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.akodad.tp4akodadrag.assistant.Assistant;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 3 : routage.
 * Deux documents sont enregistrés dans deux EmbeddingStores différents :
 * un support de cours sur l'IA (rag.pdf) et un livre de recettes de cuisine (autre.pdf).
 * Un LanguageModelQueryRouter demande au LLM de choisir, pour chaque question,
 * la (ou les) source(s) à utiliser pour le RAG.
 */
public class TestRoutage {

    public static void main(String[] args) {

        // ===== Configuration du logging =====
        configureLogger();

        // ===== ChatModel (Gemini), utilisé pour répondre ET pour le routage =====
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash")
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 du RAG : ingestion des 2 documents dans 2 EmbeddingStores =====
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        EmbeddingStore<TextSegment> storeIa =
                ingererDocument("rag.pdf", documentParser, embeddingModel);
        EmbeddingStore<TextSegment> storeCuisine =
                ingererDocument("autre.pdf", documentParser, embeddingModel);

        // ===== Phase 2 du RAG : récupération avec routage =====

        // 1. Un ContentRetriever par fichier
        ContentRetriever retrieverIa = creerContentRetriever(storeIa, embeddingModel);
        ContentRetriever retrieverCuisine = creerContentRetriever(storeCuisine, embeddingModel);

        // 2. Description du contenu des sources, en langage naturel (lue par le LM)
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverIa,
                "Support de cours sur l'intelligence artificielle : fine-tuning, RAG "
                        + "(Retrieval-Augmented Generation), embeddings et modèles de langage (LLM)");
        descriptions.put(retrieverCuisine,
                "Livre de recettes de cuisine pour chaque type de cuisinier : "
                        + "plats, ingrédients, préparation et cuisson");

        // 3. QueryRouter : le LM choisit le ou les ContentRetrievers à utiliser
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

        // 4. RetrievalAugmentor avec ce QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 5. Assistant avec le RetrievalAugmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // 6. Conversation
        conversationAvec(assistant);
    }

    /**
     * Phase 1 du RAG pour un document : chargement du fichier PDF, découpage
     * en segments, calcul des embeddings et enregistrement dans un nouveau
     * magasin d'embeddings en mémoire.
     *
     * @param nomFichier     nom du fichier PDF (dans src/main/resources)
     * @param documentParser parseur utilisé pour lire le fichier
     * @param embeddingModel modèle d'embedding utilisé pour les segments
     * @return le magasin d'embeddings qui contient les segments du document
     */
    private static EmbeddingStore<TextSegment> ingererDocument(String nomFichier,
                                                               DocumentParser documentParser,
                                                               EmbeddingModel embeddingModel) {
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    /**
     * Crée un ContentRetriever pour un magasin d'embeddings :
     * 2 résultats au maximum, avec un score supérieur ou égal à 0,5.
     *
     * @param embeddingStore magasin d'embeddings dans lequel chercher
     * @param embeddingModel modèle d'embedding utilisé pour la question
     * @return le ContentRetriever
     */
    private static ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                          EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
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
        // Évite que les logs soient affichés une 2e fois par le handler du logger racine
        packageLogger.setUseParentHandlers(false);
    }
}