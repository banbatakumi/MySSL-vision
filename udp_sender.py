import socket
import json
import time


class UDPSender:
    """UDP通信でデータを送信するためのクラス"""

    def __init__(self, ip, port):
        """
        UDPソケットを初期化する。

        Args:
            ip (str): 送信先のIPアドレス。
            port (int): 送信先のポート番号。
        """
        self.ip = ip
        self.port = port
        self.sock = None
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"UDP socket created for sending to {self.ip}:{self.port}")
        except socket.error as e:
            print(f"Failed to create UDP socket: {e}")
            self.sock = None  # ソケット作成失敗

    def send_data(self, data):
        """
        指定されたデータをJSON形式に変換してUDPで送信する。

        Args:
            data (dict): 送信するデータ（辞書形式）。
        """
        if self.sock:
            try:
                # データをJSON文字列に変換し、UTF-8バイト列にする
                json_data = json.dumps(data)
                byte_data = json_data.encode('utf-8')
                self.sock.sendto(byte_data, (self.ip, self.port))
                # print(f"Sent UDP data: {json_data}") # デバッグ用出力（必要なら有効化）
            except socket.error as e:
                # print(f"Failed to send UDP data: {e}") # エラー頻発時はコメントアウト推奨
                pass  # 送信エラーは無視して継続することが多い
            except Exception as e:
                print(f"Error during UDP send preparation: {e}")

    def close(self):
        """UDPソケットを閉じる"""
        if self.sock:
            self.sock.close()
            print("UDP socket closed.")
            self.sock = None
