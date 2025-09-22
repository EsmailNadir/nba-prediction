package com.example.demo.repository;
import com.example.demo.entity.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    List<Prediction> findByPlayerName(String playerName);
    List<Prediction> findTop10ByOrderByCreatedAtDesc();
}