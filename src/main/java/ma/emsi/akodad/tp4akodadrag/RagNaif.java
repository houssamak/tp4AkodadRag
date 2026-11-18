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

/**
 * Test 1 : RAG naïf, avec toutes les étapes du RAG écrites explicitement
 * (contrairement au RAG "facile" où elles sont cachées).
 * Le document utilisé est le support de cours sur le Fine Tuning et le RAG.
 */
public class RagNaif {

    public static void main(String[] args) {

        // ===== Création du ChatModel (Gemini), comme dans les TP précédents =====
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.8-flash")
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

        // 4. Question posée à l'assistant sur le contenu du PDF
        String question = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        String reponse = assistant.chat(question);
        System.out.println("Question : " + question);
        System.out.println("Réponse : " + reponse);
    }
}