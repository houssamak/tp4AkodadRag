package ma.emsi.akodad.tp4akodadrag.assistant;

/**
 * Assistant qui dialogue avec l'utilisateur.
 * L'implémentation est générée par LangChain4j (AiServices).
 * Interface partagée par tous les tests du TP (RAG naïf, routage, etc.).
 */
public interface Assistant {

    /**
     * Envoie un message de l'utilisateur au LLM et retourne sa réponse.
     *
     * @param userMessage le message (la question) de l'utilisateur
     * @return la réponse du LLM
     */
    String chat(String userMessage);
}