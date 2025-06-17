import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public class DistributedAISystem {
    
    // Configuración del sistema
    private static final int BASE_PORT = 8000;
    private static final int WEB_PORT_OFFSET = 100;
    private static final int HEARTBEAT_INTERVAL = 1000;
    private static final int ELECTION_TIMEOUT = 3000;
    
    // Clase principal para el nodo Worker
    static class WorkerNode {
        private int nodeId;
        private int port;
        private int webPort;
        private List<String> peers;
        private RaftConsensus raftModule;
        private AIModelManager modelManager;
        private WebServer webServer;
        private boolean running;
        
        public WorkerNode(int nodeId, List<String> peers) {
            this.nodeId = nodeId;
            this.port = BASE_PORT + nodeId;
            this.webPort = BASE_PORT + WEB_PORT_OFFSET + nodeId;
            this.peers = peers;
            this.raftModule = new RaftConsensus(nodeId, peers);
            this.modelManager = new AIModelManager();
            this.webServer = new WebServer(webPort, this);
            this.running = true;
        }
        
        public void start() {
            new Thread(this::startSocketServer).start();
            new Thread(webServer::start).start();
            new Thread(raftModule::start).start();
            
            System.out.println("Worker " + nodeId + " iniciado en puerto " + port + 
                             " (Web: " + webPort + ")");
        }
        
        private void startSocketServer() {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                while (running) {
                    Socket clientSocket = serverSocket.accept();
                    new Thread(() -> handleClient(clientSocket)).start();
                }
            } catch (IOException e) {
                System.err.println("Error en servidor Worker " + nodeId + ": " + e.getMessage());
            }
        }
        
        private void handleClient(Socket clientSocket) {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                 PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {
                
                String request = in.readLine();
                if (request == null) return;
                
                String[] parts = request.split("\\|");
                String command = parts[0];
                String response = "";
                
                switch (command) {
                    case "TRAIN":
                        if (raftModule.isLeader()) {
                            String inputs = parts[1];
                            String outputs = parts[2];
                            String modelId = trainModel(inputs, outputs);
                            response = "TRAINED|" + modelId;
                        } else {
                            String leader = raftModule.getCurrentLeader();
                            if (leader.isEmpty()) {
                                response = "NO_LEADER|Esperando elección de líder";
                            } else {
                                response = "NOT_LEADER|" + leader;
                            }
                        }
                        break;
                        
                    case "PREDICT":
                        if (raftModule.isLeader()) {
                            String modelId = parts[1];
                            String input = parts[2];
                            String prediction = predictModel(modelId, input);
                            response = "PREDICTION: " + prediction;
                        } else {
                            String leader = raftModule.getCurrentLeader();
                            if (leader.isEmpty()) {
                                response = "NO_LEADER|Esperando elección de líder";
                            } else {
                                response = "NOT_LEADER|" + leader;
                            }
                        }
                        break;
                        
                    case "LIST_MODELS":
                        response = "MODELS|" + String.join(",", modelManager.getModelIds());
                        break;
                        
                    case "STATUS":
                        response = "STATUS|" + (raftModule.isLeader() ? "LEADER" : "FOLLOWER") + 
                                  "|" + modelManager.getModelCount();
                        break;
                }
                
                out.println(response);
            } catch (IOException e) {
                System.err.println("Error manejando cliente: " + e.getMessage());
            }
        }
        
        private String trainModel(String inputs, String outputs) {
            String modelId = "MODEL_" + System.currentTimeMillis();
            
            // Simular entrenamiento distribuido
            ExecutorService executor = Executors.newFixedThreadPool(3);
            List<Future<Void>> futures = new ArrayList<>();
            
            String[] inputArray = inputs.split(",");
            String[] outputArray = outputs.split(",");
            
            for (int i = 0; i < inputArray.length; i++) {
                final int index = i;
                futures.add(executor.submit(() -> {
                    // Simular entrenamiento
                    try {
                        Thread.sleep(100); // Simular tiempo de entrenamiento
                        modelManager.addTrainingData(modelId, inputArray[index], outputArray[index]);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    return null;
                }));
            }
            
            // Esperar que termine el entrenamiento
            futures.forEach(f -> {
                try {
                    f.get();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            
            executor.shutdown();
            
            // Replicar modelo usando RAFT
            raftModule.replicateModel(modelId, inputs, outputs);
            
            System.out.println("Modelo " + modelId + " entrenado con " + inputArray.length + " muestras");
            return modelId;
        }
        
        private String predictModel(String modelId, String input) {
            return modelManager.predict(modelId, input);
        }
    }
    
    // Módulo de consenso RAFT simplificado
    static class RaftConsensus {
        private int nodeId;
        private List<String> peers;
        private volatile String currentLeader;
        private volatile boolean isLeader;
        private AtomicLong currentTerm;
        private AtomicInteger votedFor;
        private long lastHeartbeat;
        private Random random;
        private boolean electionInProgress;
        private ServerSocket voteSocket;
        
        public RaftConsensus(int nodeId, List<String> peers) {
            this.nodeId = nodeId;
            this.peers = peers;
            this.currentTerm = new AtomicLong(0);
            this.votedFor = new AtomicInteger(-1);
            this.lastHeartbeat = System.currentTimeMillis();
            this.random = new Random();
            this.currentLeader = "";
            this.electionInProgress = false;
        }
        
        public void start() {
            // Iniciar servidor para recibir votos
            new Thread(this::startVoteServer).start();
            // Iniciar proceso de elección
            new Thread(this::runElectionProcess).start();
        }
        
        private void startVoteServer() {
            try {
                voteSocket = new ServerSocket(BASE_PORT + 200 + nodeId);
                while (true) {
                    Socket clientSocket = voteSocket.accept();
                    new Thread(() -> handleVoteRequest(clientSocket)).start();
                }
            } catch (IOException e) {
                System.err.println("Error en servidor de votos: " + e.getMessage());
            }
        }
        
        private void handleVoteRequest(Socket socket) {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {
                
                String request = in.readLine();
                String[] parts = request.split("\\|");
                
                if ("VOTE_REQUEST".equals(parts[0])) {
                    int candidateId = Integer.parseInt(parts[1]);
                    long candidateTerm = Long.parseLong(parts[2]);
                    
                    // Votar si no hemos votado en este término o si es un término superior
                    if (candidateTerm > currentTerm.get() || 
                        (candidateTerm == currentTerm.get() && votedFor.get() == -1)) {
                        
                        currentTerm.set(candidateTerm);
                        votedFor.set(candidateId);
                        isLeader = false;
                        currentLeader = "";
                        
                        out.println("VOTE_GRANTED|" + nodeId);
                        System.out.println("Nodo " + nodeId + " votó por nodo " + candidateId + 
                                         " en término " + candidateTerm);
                    } else {
                        out.println("VOTE_DENIED|" + nodeId);
                    }
                }
                
                if ("HEARTBEAT".equals(parts[0])) {
                    int leaderId = Integer.parseInt(parts[1]);
                    long leaderTerm = Long.parseLong(parts[2]);
                    
                    if (leaderTerm >= currentTerm.get()) {
                        currentTerm.set(leaderTerm);
                        currentLeader = "NODE_" + leaderId;
                        lastHeartbeat = System.currentTimeMillis();
                        isLeader = false;
                        out.println("HEARTBEAT_ACK|" + nodeId);
                    }
                }
                
            } catch (IOException e) {
                System.err.println("Error manejando voto: " + e.getMessage());
            }
        }
        
        private void runElectionProcess() {
            // Delay inicial aleatorio para evitar elecciones simultáneas
            try {
                Thread.sleep(2000 + random.nextInt(3000));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            
            while (true) {
                try {
                    // Verificar si necesitamos iniciar elección
                    if (!isLeader && !electionInProgress && 
                        (System.currentTimeMillis() - lastHeartbeat) > ELECTION_TIMEOUT) {
                        startElection();
                    }
                    
                    // Si somos líder, enviar heartbeats
                    if (isLeader) {
                        sendHeartbeats();
                    }
                    
                    Thread.sleep(HEARTBEAT_INTERVAL);
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
        
        private void startElection() {
            electionInProgress = true;
            currentTerm.incrementAndGet();
            votedFor.set(nodeId);
            isLeader = false;
            currentLeader = "";
            
            System.out.println("Nodo " + nodeId + " iniciando elección para término " + currentTerm.get());
            
            int votes = 1; // Voto propio
            
            // Solicitar votos a otros nodos
            for (int i = 1; i <= 3; i++) {
                if (i != nodeId) {
                    if (requestVote(i)) {
                        votes++;
                    }
                }
            }
            
            // Verificar si ganamos la elección (mayoría)
            if (votes > 3 / 2) { // Mayoría de 3 nodos
                becomeLeader();
            } else {
                System.out.println("Nodo " + nodeId + " perdió la elección (" + votes + " votos)");
            }
            
            electionInProgress = false;
        }
        
        private boolean requestVote(int targetNodeId) {
            try (Socket socket = new Socket("localhost", BASE_PORT + 200 + targetNodeId);
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                
                out.println("VOTE_REQUEST|" + nodeId + "|" + currentTerm.get());
                String response = in.readLine();
                
                return response != null && response.startsWith("VOTE_GRANTED");
                
            } catch (IOException e) {
                // Nodo no disponible
                return false;
            }
        }
        
        private void becomeLeader() {
            isLeader = true;
            currentLeader = "NODE_" + nodeId;
            lastHeartbeat = System.currentTimeMillis();
            System.out.println("*** Nodo " + nodeId + " es ahora LÍDER para término " + currentTerm.get() + " ***");
        }
        
        private void sendHeartbeats() {
            for (int i = 1; i <= 3; i++) {
                if (i != nodeId) {
                    sendHeartbeat(i);
                }
            }
        }
        
        private void sendHeartbeat(int targetNodeId) {
            try (Socket socket = new Socket("localhost", BASE_PORT + 200 + targetNodeId);
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                
                out.println("HEARTBEAT|" + nodeId + "|" + currentTerm.get());
                in.readLine(); // Leer ACK
                
            } catch (IOException e) {
                // Nodo no disponible
            }
        }
        
        public void replicateModel(String modelId, String inputs, String outputs) {
            if (isLeader) {
                System.out.println("Replicando modelo " + modelId + " a todos los nodos");
                // En implementación real, esto enviaría el modelo a los followers
            }
        }
        
        public boolean isLeader() { return isLeader; }
        public String getCurrentLeader() { return currentLeader; }
    }
    
    // Gestor de modelos de IA
    static class AIModelManager {
        private Map<String, AIModel> models;
        private ReentrantLock lock;
        
        public AIModelManager() {
            this.models = new ConcurrentHashMap<>();
            this.lock = new ReentrantLock();
        }
        
        public void addTrainingData(String modelId, String input, String output) {
            lock.lock();
            try {
                models.computeIfAbsent(modelId, k -> new AIModel(k)).addTrainingData(input, output);
            } finally {
                lock.unlock();
            }
        }
        
        public String predict(String modelId, String input) {
            AIModel model = models.get(modelId);
            if (model != null) {
                return model.predict(input);
            }
            return "ERROR: Modelo no encontrado";
        }
        
        public Set<String> getModelIds() {
            return models.keySet();
        }
        
        public int getModelCount() {
            return models.size();
        }
    }
    
    // Modelo de IA simplificado
    static class AIModel {
        private String modelId;
        private Map<String, String> trainingData;
        
        public AIModel(String modelId) {
            this.modelId = modelId;
            this.trainingData = new ConcurrentHashMap<>();
        }
        
        public void addTrainingData(String input, String output) {
            trainingData.put(input, output);
        }
        
        public String predict(String input) {
            // Predicción simple basada en datos de entrenamiento
            if (trainingData.containsKey(input)) {
                return trainingData.get(input);
            }
            
            // Predicción básica para datos no vistos
            try {
                double inputNum = Double.parseDouble(input);
                return String.valueOf(inputNum * 2); // Función simple
            } catch (NumberFormatException e) {
                return "PREDICTION_" + input.hashCode();
            }
        }
    }
    
    // Servidor web para monitoreo
    static class WebServer {
        private int port;
        private WorkerNode node;
        
        public WebServer(int port, WorkerNode node) {
            this.port = port;
            this.node = node;
        }
        
        public void start() {
            try (ServerSocket serverSocket = new ServerSocket(port)) {
                System.out.println("Servidor web iniciado en puerto " + port);
                
                while (true) {
                    Socket clientSocket = serverSocket.accept();
                    new Thread(() -> handleWebRequest(clientSocket)).start();
                }
            } catch (IOException e) {
                System.err.println("Error en servidor web: " + e.getMessage());
            }
        }
        
        private void handleWebRequest(Socket clientSocket) {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                 PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {
                
                String request = in.readLine();
                
                out.println("HTTP/1.1 200 OK");
                out.println("Content-Type: text/html");
                out.println();
                out.println("<html><body>");
                out.println("<h1>Worker Node " + node.nodeId + " Monitor</h1>");
                out.println("<p>Estado: " + (node.raftModule.isLeader() ? "LÍDER" : "SEGUIDOR") + "</p>");
                out.println("<p>Modelos: " + node.modelManager.getModelCount() + "</p>");
                out.println("<p>Término actual: " + node.raftModule.currentTerm.get() + "</p>");
                out.println("<h2>Modelos disponibles:</h2>");
                out.println("<ul>");
                for (String modelId : node.modelManager.getModelIds()) {
                    out.println("<li>" + modelId + "</li>");
                }
                out.println("</ul>");
                out.println("</body></html>");
                
            } catch (IOException e) {
                System.err.println("Error manejando request web: " + e.getMessage());
            }
        }
    }
    
    // Cliente para pruebas
    static class TestClient {
        private String serverHost;
        private int serverPort;
        private String lastModelId = "";
        
        public TestClient(String serverHost, int serverPort) {
            this.serverHost = serverHost;
            this.serverPort = serverPort;
        }
        
        public void trainModel(String inputs, String outputs) {
            String response = sendRequest("TRAIN|" + inputs + "|" + outputs);
            if (response.startsWith("TRAINED|")) {
                lastModelId = response.split("\\|")[1];
            }
        }
        
        public void predictModel(String modelId, String input) {
            sendRequest("PREDICT|" + modelId + "|" + input);
        }
        
        public void predictWithKnownModel(String input) {
            if (!lastModelId.isEmpty()) {
                sendRequest("PREDICT|" + lastModelId + "|" + input);
            } else {
                System.out.println("No hay modelo entrenado disponible");
            }
        }
        
        public void listModels() {
            sendRequest("LIST_MODELS");
        }
        
        public void getStatus() {
            sendRequest("STATUS");
        }
        
        private String sendRequest(String request) {
            try (Socket socket = new Socket(serverHost, serverPort);
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                
                out.println(request);
                String response = in.readLine();
                System.out.println("Respuesta: " + response);
                return response;
                
            } catch (IOException e) {
                System.err.println("Error conectando al servidor: " + e.getMessage());
                return "";
            }
        }
    }
    
    // Método principal
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Uso: java DistributedAISystem [worker|client] [parámetros]");
            System.out.println("Worker: java DistributedAISystem worker <nodeId>");
            System.out.println("Client: java DistributedAISystem client <serverHost> <serverPort>");
            return;
        }
        
        String mode = args[0];
        
        if ("worker".equals(mode)) {
            int nodeId = Integer.parseInt(args[1]);
            List<String> peers = Arrays.asList("localhost:8001", "localhost:8002", "localhost:8003");
            
            WorkerNode worker = new WorkerNode(nodeId, peers);
            worker.start();
            
            // Mantener el proceso activo
            try {
                Thread.sleep(Long.MAX_VALUE);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
        } else if ("client".equals(mode)) {
            String serverHost = args[1];
            int serverPort = Integer.parseInt(args[2]);
            
            TestClient client = new TestClient(serverHost, serverPort);
            
            // Pruebas del cliente
            System.out.println("=== PRUEBAS DEL CLIENTE ===");
            
            // Entrenar modelo
            System.out.println("1. Entrenando modelo...");
            client.trainModel("1,2,3,4,5", "2,4,6,8,10");
            
            // Esperar un poco
            try { Thread.sleep(3000); } catch (InterruptedException e) {}
            
            // Listar modelos
            System.out.println("2. Listando modelos...");
            client.listModels();
            
            // Hacer predicción con dato conocido
            System.out.println("3. Haciendo predicción con dato conocido...");
            System.out.println("Dato: 3");
            client.predictWithKnownModel("3");
            
            // Hacer predicción con dato nuevo
            System.out.println("4. Haciendo predicción con dato nuevo...");
            System.out.println("Dato: 6");
            client.predictWithKnownModel("6");
            
            // Verificar estado
            System.out.println("5. Verificando estado...");
            client.getStatus();
        }
    }
}