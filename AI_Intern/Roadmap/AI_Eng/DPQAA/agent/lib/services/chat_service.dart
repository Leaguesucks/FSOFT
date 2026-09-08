import 'dart:convert';

import 'package:http/http.dart' as http;

class ChatService {
  final String baseUrl = "http://127.0.0.1:8000";

  final http.Client _client = http.Client();

  Future<void> sendMessage(
    String text,
    void Function(String chunk) onChunk,
  ) async {
    final request = http.Request(
      "POST",
      Uri.parse("$baseUrl/chat"),
    );

    request.headers["Content-Type"] =
        "application/json";

    request.body = jsonEncode({
      "text": text,
      "chat_type": "plain_text",
    });

    final response = await _client.send(request);

    if (response.statusCode != 200) {
      throw Exception(
        "Server returned ${response.statusCode}",
      );
    }

    await for (
      final chunk
      in response.stream.transform(utf8.decoder)
    ) {
      onChunk(chunk);
    }
  }

  void dispose() {
    _client.close();
  }
}