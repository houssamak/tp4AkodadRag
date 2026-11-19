package ma.emsi.akodad.tp4akodadrag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.akodad.tp4akodadrag.assistant.Assistant;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 1 : RAG naïf, avec toutes les étapes du RAG écrites explicitement
 * (contrairement au RAG "facile" où elles sont cachées).
 * Le document utilisé est le support de cours sur le Fine Tuning et le RAG.
 * Le logging permet de voir les requêtes envoyées au LLM et ses réponses.
 */
public class RagNaif {

    public static void main(String[] args) {

        // ===== Configuration du logging (au début du code) =====
        configureLogger();

        // ===== Création du ChatModel (Gemini), comme dans les TP précédents =====
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.8-flash")
                // Affiche dans les logs les requêtes envoyées au LLM et ses réponses
                .logRequestsAndResponses(true)
                .build();

        // ================= Phase 1 du RAG : enregistrement des embeddings =================

        // 1. Création d'un Document à partir du fichier PDF (parseur Apache Tika)
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        // 2. Découpage du document en morceaux (TextSegment)
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // 3. Création du modèle d'embedding
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        // 4. Création des embeddings des segments
        //    embedAll retourne une Response<List<Embedding>> ; content() donne la liste
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // 5. Ajout des embeddings et des segments associés dans un magasin en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ================= Phase 2 du RAG : utilisation des embeddings =================

        // 1. Création du ContentRetriever : 2 résultats max, score >= 0,5
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 2. Mémoire pour 10 messages
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // 3. Création de l'assistant, avec le ContentRetriever en plus
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // ===== Version 1 (étape 3.a) : une seule question =====
        // String question = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        // String reponse = assistant.chat(question);
        // System.out.println("Question : " + question);
        // System.out.println("Réponse : " + reponse);

        // 4. Conversation : plusieurs questions sans recompiler
        conversationAvec(assistant);
    }

    /**
     * Permet de poser plusieurs questions à l'assistant, saisies au clavier,
     * sans avoir à recompiler. La conversation s'arrête quand on tape "fin".
     * Si l'API du LLM renvoie une erreur (par exemple 503 si le modèle est
     * surchargé), l'erreur est affichée et la conversation continue.
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
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}