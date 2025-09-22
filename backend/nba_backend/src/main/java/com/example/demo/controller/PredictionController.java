package com.example.demo.controller;

import com.example.demo.entity.Prediction;
import com.example.demo.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.Map;
import java.util.List;
import java.util.HashMap;

@RestController
@RequestMapping("/api/predictions")
@CrossOrigin(origins = "*")
public class PredictionController {

    @Autowired
    private PredictionService predictionService;
    
    @Autowired
    private RestTemplate restTemplate;

    // Create a new prediction
    @PostMapping("/create")
    public ResponseEntity<?> createPrediction(@RequestBody Map<String, String> request) {
        String playerName = request.get("playerName");
        Object prediction = predictionService.createPrediction(playerName);
        return ResponseEntity.ok(prediction);
    }

    // Get recent predictions
    @GetMapping("/recent")
    public ResponseEntity<List<Prediction>> getRecentPredictions() {
        List<Prediction> predictions = predictionService.getRecentPredictions();
        return ResponseEntity.ok(predictions);
    }

    // Get predictions for specific player
    @GetMapping("/player/{playerName}")
    public ResponseEntity<List<Prediction>> getPlayerHistory(@PathVariable String playerName) {
        List<Prediction> predictions = predictionService.getPlayerHistory(playerName);
        return ResponseEntity.ok(predictions);
    }

    // Get all available players
    @GetMapping("/players")
    public ResponseEntity<?> getAllPlayers() {
        try {
            ResponseEntity<Object[]> response = restTemplate.getForEntity(
                "http://localhost:5004/players",
                Object[].class
            );
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", "Failed to get players: " + e.getMessage());
            return ResponseEntity.status(500).body(error);
        }
    }

    // Test endpoint
    @GetMapping("/test")
    public String test() {
        return "Predictions API is working!";
    }
}