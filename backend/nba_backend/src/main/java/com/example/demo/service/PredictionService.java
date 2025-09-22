package com.example.demo.service;

import com.example.demo.entity.Prediction;
import com.example.demo.repository.PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

@Service
public class PredictionService {

    @Autowired
    private PredictionRepository repository;

    @Autowired
    private RestTemplate restTemplate;

    private final String PYTHON_URL = "http://localhost:5004";

    // Method 1: Create prediction
    public Object createPrediction(String playerName) {

        // "make sure the player is active and exists in the app"
        if (playerName == null || playerName.trim().isEmpty()) {
            throw new IllegalArgumentException("Player name cannot be empty");
        }

        // Call Python service with correct parameter name
        Map<String, Object> pythonRequest = new HashMap<>();
        pythonRequest.put("playerName", playerName);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(pythonRequest, headers);

        try {
            // Call Python service
            ResponseEntity<Map> response = restTemplate.postForEntity(
                    PYTHON_URL + "/predict",
                    entity,
                    Map.class
            );

            Map<String, Object> result = response.getBody();

            // Create prediction entity
            Prediction prediction = new Prediction();
            
            // Get player name from response (cleaned/corrected name)
            String actualPlayerName = result.get("playerName") != null ? 
                result.get("playerName").toString() : playerName;
            prediction.setPlayerName(actualPlayerName);

            // Get predicted points
            Object predictedPointsObj = result.get("predictedPoints");
            if (predictedPointsObj != null) {
                Double points = Double.parseDouble(predictedPointsObj.toString());
                prediction.setPredictedPoints(points);
            }

            // Get confidence
            Object confidenceObj = result.get("confidence");
            if (confidenceObj != null) {
                Double conf = Double.parseDouble(confidenceObj.toString());
                prediction.setConfidence(conf);
            }

            // Set timestamps
            prediction.setCreatedAt(LocalDateTime.now());
            prediction.setGameDate(LocalDateTime.now()); // For now, using current time

            // Save to database
            Prediction savedPrediction = repository.save(prediction);
            
            // Return enhanced response with all Python service data
            Map<String, Object> enhancedResponse = new HashMap<>(result);
            enhancedResponse.put("id", savedPrediction.getId());
            enhancedResponse.put("createdAt", savedPrediction.getCreatedAt().toString());
            enhancedResponse.put("gameDate", savedPrediction.getGameDate().toString());
            
            return enhancedResponse;

        } catch (Exception e) {
            // "throw errors if something doesnt work"
            throw new RuntimeException("Failed to get prediction: " + e.getMessage());
        }
    }

    // Method 2: Get recent predictions (MOVED OUTSIDE)
    public List<Prediction> getRecentPredictions() {
        return repository.findTop10ByOrderByCreatedAtDesc();
    }

    // Method 3: Get player history (MOVED OUTSIDE)
    public List<Prediction> getPlayerHistory(String playerName) {
        return repository.findByPlayerName(playerName);
    }

}